import os
import io
import json
import torch
import logging
import pandas as pd
import numpy as np

from abodybuilder3.utils import string_to_input, add_atom37_to_output
from abodybuilder3.lightning_module import LitABB3
from abodybuilder3.language_model import ProtT5
from abodybuilder3.openfold.np.protein import Protein, to_pdb
from abodybuilder3.openfold.np.relax.cleanup import fix_pdb

# ================= USER CONFIGURATION =================
CSV_FILE = "antibodies.csv"
MODEL_TYPE = "abodybuilder3-lm"  # Options: "abodybuilder3", "abodybuilder3-lm"
USE_PRECOMPUTED = True  # Only valid if using abodybuilder3-lm
PRECOMPUTED_EMBEDDING_DIR = "data/structures/structures_plm"
MODEL_PATH = {
    "abodybuilder3": "plddt-loss/best_second_stage.ckpt",
    "abodybuilder3-lm": "language-loss/best_second_stage.ckpt"
}[MODEL_TYPE]
OUTPUT_DIR = "abodybuilder3"
REPEATS = 3
# =====================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def output_to_pdb(output: dict, model_input: dict, b_factors=None) -> str:
    aatype = model_input["aatype"].squeeze().cpu().numpy().astype(int)
    atom37 = output["atom37"]
    chain_index = 1 - model_input["is_heavy"].cpu().numpy().astype(int)
    atom_mask = output["atom37_atom_exists"].cpu().numpy().astype(int)
    residue_index = np.arange(len(atom37))
    if b_factors is None:
        b_factors = np.zeros_like(atom_mask)
    protein = Protein(
        aatype=aatype,
        atom_positions=atom37,
        atom_mask=atom_mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index,
    )
    return fix_pdb(io.StringIO(to_pdb(protein)), {})

def compute_plddt(plddt: torch.Tensor) -> torch.Tensor:
    pdf = torch.nn.functional.softmax(plddt, dim=-1)
    vbins = torch.arange(1, 101, 2).to(plddt.device).float()
    return pdf @ vbins

def predict_structure(name, vh, vl, model, model_type, repeat_index, device):
    subdir = os.path.join(OUTPUT_DIR, f"{model_type}_{repeat_index}")
    os.makedirs(subdir, exist_ok=True)
    pdb_path = os.path.join(subdir, f"{name}_{model_type}_{repeat_index}.pdb")

    ab_input = string_to_input(heavy=vh, light=vl)

    if model_type == "abodybuilder3-lm":
        if USE_PRECOMPUTED:
            pt_file = os.path.join(PRECOMPUTED_EMBEDDING_DIR, f"{name}.pt")
            embedding = torch.load(pt_file)["plm_embedding"]
            logging.info(f"Loaded precomputed embedding for {name}")
        else:
            protT5 = ProtT5()
            embedding = protT5.get_embeddings([vh], [vl]).to(device)
            logging.info(f"Generated embedding for {name}")
        ab_input["single"] = embedding.unsqueeze(0)

    ab_input_batch = {
        key: (value.unsqueeze(0).to(device) if key not in ["single", "pair"]
              else value.to(device))
        for key, value in ab_input.items()
    }

    model.eval()
    model.to(device)

    with torch.no_grad():
        output = model(ab_input_batch, ab_input_batch["aatype"])
        output = add_atom37_to_output(output, ab_input["aatype"].to(device))

    plddt = compute_plddt(output["plddt"]).squeeze().detach().cpu().numpy()
    b_factors = np.expand_dims(plddt, 1).repeat(37, 1)
    pdb_string = output_to_pdb(output, ab_input, b_factors)

    with open(pdb_path, "w") as f:
        f.write(pdb_string)

    logging.info(f"{name} rep {repeat_index} written to {pdb_path}")

def main():
    df = pd.read_csv(CSV_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loaded model from {MODEL_PATH}")
    module = LitABB3.load_from_checkpoint(MODEL_PATH)
    model = module.model

    for _, row in df.iterrows():
        name, vh, vl = row["name"], row["vh"], row["vl"]
        for rep in range(1, REPEATS + 1):
            try:
                predict_structure(name, vh, vl, model, MODEL_TYPE, rep, device)
            except Exception as e:
                logging.error(f"Failed {name} rep {rep}: {e}")

if __name__ == "__main__":
    main()
