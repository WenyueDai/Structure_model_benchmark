# STEP 3: SASA + fluctuation analysis

!pip install pandas biopython freesasa plotly tqdm

import os
import freesasa
import pandas as pd
import plotly.express as px
from Bio import PDB
from tqdm import tqdm
from scipy.stats import pearsonr

def load_structure(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure('structure', pdb_path)

def calculate_relative_sasa(structure):
    io = PDB.PDBIO()
    io.set_structure(structure)
    temp_path = "temp_structure.pdb"
    io.save(temp_path)

    structure_sasa = freesasa.Structure(temp_path)
    result = freesasa.calc(structure_sasa)

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', temp_path)

    sasa_per_residue = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):
                    resname = residue.get_resname()
                    res_id = residue.id[1]
                    chain_id = chain.id
                    if resname == "MET":
                        sasa = result.residueAreas().get((chain_id, resname, res_id), 0.0)
                        sasa_per_residue[f"{chain_id}{res_id}"] = sasa / 160.0
    os.remove(temp_path)
    return sasa_per_residue

base_folder = "renumbered_output"
models = ["igfold", "immunebuilder"]
repeats = ["1", "2", "3"]

csv_path = "antibody_sequences.csv"
df = pd.read_csv(csv_path)

all_sasa = {}
for idx, row in tqdm(df.iterrows(), total=len(df)):
    name = row['name']
    vh_seq = row['vh_sequence']
    vl_seq = row['vl_sequence']
    residue_records = {}

    for model_name in models:
        for rep in repeats:
            folder = os.path.join(base_folder, f"{model_name}_{rep}")
            pdb_file = os.path.join(folder, f"{name}_{model_name}_{rep}_renumbered.pdb")
            if os.path.exists(pdb_file):
                try:
                    structure = load_structure(pdb_file)
                    sasa_dict = calculate_relative_sasa(structure)
                    for res, sasa in sasa_dict.items():
                        if res not in residue_records:
                            residue_records[res] = {}
                        residue_records[res][f"{model_name}_{rep}"] = sasa
                except Exception as e:
                    print(f"Error processing {pdb_file}: {e}")
    all_sasa[name] = residue_records

records = []
for antibody, residue_data in all_sasa.items():
    for residue, model_sasa in residue_data.items():
        row = {"antibody": antibody, "chain_res": residue}
        row.update(model_sasa)
        records.append(row)

sasa_df = pd.DataFrame(records)
for model in models:
    cols = [f"{model}_{r}" for r in repeats]
    sasa_df[f"{model}_mean"] = sasa_df[cols].mean(axis=1)
    sasa_df[f"{model}_std"] = sasa_df[cols].std(axis=1)

sasa_df["avg_rela_sasa"] = sasa_df[[f"{m}_mean" for m in models]].mean(axis=1)
sasa_df["sasa_std"] = sasa_df[[f"{m}_mean" for m in models]].std(axis=1)
sasa_df["sasa_range"] = sasa_df[[f"{m}_mean" for m in models]].max(axis=1) - sasa_df[[f"{m}_mean" for m in models]].min(axis=1)
sasa_df["stability_label"] = sasa_df["sasa_std"].apply(lambda x: "unstable" if x > 0.15 else "stable")

# Plot
fig = px.imshow(sasa_df[[f"{m}_mean" for m in models]].corr().applymap(lambda r: round(r**2, 3)),
                title="SASA R² Matrix", text_auto=True, color_continuous_scale="Blues")
fig.show()

sasa_df.to_csv("sasa_summary_with_model_fluctuations.csv", index=False)
print("SASA summary table saved")
