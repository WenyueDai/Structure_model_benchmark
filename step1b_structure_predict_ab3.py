import pandas as pd
import torch
import os
import json
import uuid
import logging
import numpy as np
import io

from abodybuilder3.utils import string_to_input, add_atom37_to_output
from abodybuilder3.lightning_module import LitABB3
from abodybuilder3.openfold.np.protein import Protein, to_pdb
from abodybuilder3.openfold.np.relax.cleanup import fix_pdb
from abodybuilder3.language_model import ProtT5

# ==== CONFIGURATION =====
CSV_FILE = "antibodies.csv"  # <-- path to your input CSV
MODEL_TYPE = "abodybuilder3-lm"  # options: "abodybuilder3" or "abodybuilder3-lm"
MODEL_PATH = "language-loss/best_second_stage.ckpt"  # or plddt-loss/... for abodybuilder3
OUTPUT_DIR = "output"
REPEATS = 3
USE_PRECOMPUTED = False  # only used for abodybuilder3-lm
# ========================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_seq(seq):
    return ''.join([aa for aa in seq.upper() if aa in VALID_AA])


def compute_plddt(plddt: torch.Tensor) -> torch.Tensor:
    pdf = torch.nn.functional.softmax(plddt, dim=-1)
    vbins = torch.arange(1, 101, 2).to(plddt.device).float()
    return pdf @ vbins


def output_to_pdb(output: dict, model_input: dict) -> str:
    aatype = model_input["aatype"].squeeze().cpu().numpy().astype(int)
    atom37 = output["atom37"]
    chain_index = 1 - model_input["is_heavy"].cpu().numpy().astype(int)
    atom_mask = output["atom37_atom_exists"].cpu().numpy().astype(int)
    residue_index = np.arange(len(atom37))
    protein = Protein(
        aatype=aatype,
        atom_positions=atom37,
        atom_mask=atom_mask,
        residue_index=residue_index,
        b_factors=np.zeros_like(atom_mask),
        chain_index=chain_index,
    )
    return fix_pdb(io.StringIO(to_pdb(protein)), {})


def run_model(vh, vl, model, device, use_language_model=False):
    vh, vl = clean_seq(vh), clean_seq(vl)
    ab_input = string_to_input(heavy=vh, light=vl)

    if use_language_model:
        if USE_PRECOMPUTED:
            embedding = torch.load("your_precomputed_tensor.pt").to(device)
        else:
            protT5 = ProtT5()
            embedding = protT5.get_embeddings([vh], [vl]).to(device)

        # Trim everything to match min length
        min_len = min(ab_input["single"].shape[0], embedding.shape[0])
        ab_input["single"] = embedding[:min_len].unsqueeze(0)
        ab_input["aatype"] = ab_input["aatype"][:min_len]
        ab_input["residue_index"] = ab_input["residue_index"][:min_len]
        ab_input["is_heavy"] = ab_input["is_heavy"][:min_len]
        ab_input["pair"] = ab_input["pair"][:min_len, :min_len, :]

    ab_input_batch = {
        k: (v.unsqueeze(0).to(device) if k not in ["single", "pair"] else v.to(device))
        for k, v in ab_input.items()
    }

    model.eval()
    with torch.no_grad():
        output = model(ab_input_batch, ab_input_batch["aatype"])
        output = add_atom37_to_output(output, ab_input["aatype"].to(device))

    if "plddt" in output:
        plddt = compute_plddt(output["plddt"]).squeeze().detach().cpu().numpy()
    else:
        plddt = np.zeros(len(ab_input["aatype"]), dtype=np.float32)

    aatype = ab_input["aatype"].squeeze().cpu().numpy().astype(int)
    atom37 = output["atom37"]
    chain_index = 1 - ab_input["is_heavy"].cpu().numpy().astype(int)
    atom_mask = output["atom37_atom_exists"].cpu().numpy().astype(int)
    residue_index = np.arange(len(atom37))
    b_factors = np.expand_dims(plddt, 1).repeat(37, 1)

    protein = Protein(
        aatype=aatype,
        atom_positions=atom37,
        atom_mask=atom_mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index,
    )

    return fix_pdb(io.StringIO(to_pdb(protein)), {}), plddt


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading {MODEL_TYPE} from {MODEL_PATH}")
    module = LitABB3.load_from_checkpoint(MODEL_PATH)
    model = module.model.to(device)

    use_lm = MODEL_TYPE.lower() == "abodybuilder3-lm"

    for _, row in df.iterrows():
        name, vh, vl = row["name"], row["vh"], row["vl"]

        for rep in range(1, REPEATS + 1):
            try:
                outname = f"{name}_{MODEL_TYPE}_{rep}"
                pdb_string, plddt = run_model(vh, vl, model, device, use_language_model=use_lm)

                with open(os.path.join(OUTPUT_DIR, f"{outname}.pdb"), "w") as f:
                    f.write(pdb_string)

                metrics = {
                    "name": name,
                    "repeat": rep,
                    "vh": vh,
                    "vl": vl,
                    "mean_plddt": round(float(np.mean(plddt)), 3),
                }

                with open(os.path.join(OUTPUT_DIR, f"{outname}.json"), "w") as f:
                    json.dump(metrics, f)
                    f.write("\n")

                logging.info(f"{outname} done")

            except Exception as e:
                logging.error(f"Failed {name} rep {rep}: {e}")


if __name__ == "__main__":
    main()
