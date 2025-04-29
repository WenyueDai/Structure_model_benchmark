# STEP 1: Predict Antibody Structures (with 3 repeats per model)

!pip install pandas igfold ImmuneBuilder tqdm

import pandas as pd
import os
import time
from tqdm import tqdm
from igfold import IgFoldRunner
from ImmuneBuilder import ABodyBuilder2

# Output base
output_base = "predictions_output"
os.makedirs(output_base, exist_ok=True)

# Models to run
model_configs = {
    "igfold": lambda: IgFoldRunner(),
    "immunebuilder": lambda: ABodyBuilder2()
}

# Create subfolders for each model + repeat
repeats = ["1", "2", "3"]
for model_name in model_configs:
    for rep in repeats:
        os.makedirs(os.path.join(output_base, f"{model_name}_{rep}"), exist_ok=True)

# Load CSV
csv_path = "antibody_sequences.csv"
df = pd.read_csv(csv_path)

# Prediction functions
def predict_igfold(name, vh, vl, out_path):
    runner = IgFoldRunner()
    runner.fold(output_path=out_path, sequences={"H": vh, "L": vl}, do_refine=True, do_renum=True)

def predict_immunebuilder(name, vh, vl, out_path):
    predictor = ABodyBuilder2()
    ab = predictor.predict({"H": vh, "L": vl})
    ab.save(out_path)

# Map model to function
model_functions = {
    "igfold": predict_igfold,
    "immunebuilder": predict_immunebuilder
}

# Run predictions
for idx, row in tqdm(df.iterrows(), total=len(df)):
    name = row['name']
    vh_seq = row['vh_sequence']
    vl_seq = row['vl_sequence']

    for model_name in model_configs:
        for rep in repeats:
            out_path = os.path.join(output_base, f"{model_name}_{rep}", f"{name}_{model_name}_{rep}.pdb")
            try:
                model_functions[model_name](name, vh_seq, vl_seq, out_path)
            except Exception as e:
                print(f"Error for {name} ({model_name} rep {rep}): {e}")

print("Structure prediction with 3x repeats complete.")
