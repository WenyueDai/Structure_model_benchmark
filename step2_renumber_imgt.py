# STEP 2: Renumber structures to IMGT (3 repeats per model)

!pip install pandas biopython anarci tqdm

import os
import pandas as pd
from Bio import PDB
from anarci import number
from tqdm import tqdm

three_to_one = {
    'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G',
    'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L', 'MET':'M', 'ASN':'N',
    'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S', 'THR':'T', 'VAL':'V',
    'TRP':'W', 'TYR':'Y'
}

def load_structure(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure('structure', pdb_path)

def save_structure(structure, save_path):
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(save_path)

def renumber_to_imgt(structure, vh_seq, vl_seq):
    numbering, _ = number([('H', vh_seq), ('L', vl_seq)], scheme='imgt')
    imgt_map = {}
    for (chain_id, residues) in numbering[0]:
        for (num, aa, _) in residues:
            if num is not None:
                imgt_map[(chain_id, aa)] = num

    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):
                    resname = residue.get_resname()
                    chain_id = chain.id
                    try:
                        aa_one = three_to_one[resname]
                        key = (chain_id, aa_one)
                        if key in imgt_map:
                            residue.id = (' ', imgt_map[key], ' ')
                    except KeyError:
                        continue
    return structure

csv_path = "antibody_sequences.csv"
df = pd.read_csv(csv_path)
models = ["igfold", "immunebuilder"]
repeats = ["1", "2", "3"]

for model in models:
    for rep in repeats:
        os.makedirs(f"renumbered_output/{model}_{rep}", exist_ok=True)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    name = row['name']
    vh_seq = row['vh_sequence']
    vl_seq = row['vl_sequence']

    for model in models:
        for rep in repeats:
            input_path = f"predictions_output/{model}_{rep}/{name}_{model}_{rep}.pdb"
            output_path = f"renumbered_output/{model}_{rep}/{name}_{model}_{rep}_renumbered.pdb"

            if os.path.exists(input_path):
                try:
                    structure = load_structure(input_path)
                    structure = renumber_to_imgt(structure, vh_seq, vl_seq)
                    save_structure(structure, output_path)
                except Exception as e:
                    print(f" Error renumbering {input_path}: {e}")

print("All repeated structures renumbered to IMGT format")
