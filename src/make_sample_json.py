import json
import pandas as pd
import os

TRAIN_DATA = "data/train.csv"

if os.path.exists(TRAIN_DATA):
    df = pd.read_csv(TRAIN_DATA)
    
    row = df.drop(columns=["risco_defasagem"]).iloc[0].to_dict()

    payload = {"data": row}

    with open("data/sample_request.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ Gerado novo exemplo completo em: data/sample_request.json")
else:
    print("❌ Erro: data/train.csv não encontrado. Rode o make_dataset.py primeiro.")