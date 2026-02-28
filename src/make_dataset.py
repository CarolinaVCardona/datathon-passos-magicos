import pandas as pd
from pathlib import Path

EXCEL_PATH = "data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
OUT_PATH = "data/train.csv"

def main():
    sheet = "PEDE2024" 
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)

    TARGET_COL = "Defasagem"

    if TARGET_COL not in df.columns:
        raise ValueError(f"Não encontrei a coluna '{TARGET_COL}' no dataframe.")

    df["risco_defasagem"] = (df[TARGET_COL] < 0).astype(int)

    leak_keywords = [
        "Rec ",
        "Destaque",
        "Atingiu",
        "Indicado",
        "IPV",
        "IAN",
        "IEG",
        "IPS",
        "IDA"
    ]

    cols_to_drop = []

    for col in df.columns:
        for keyword in leak_keywords:
            if keyword in col:
                cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop, errors="ignore")
    cols_leak = [c for c in ["Defasagem", "Fase ideal"] if c in df.columns]
    df = df.drop(columns=cols_leak, errors="ignore")

    cols_id = [c for c in ["RA", "Nome"] if c in df.columns]
    df = df.drop(columns=cols_id, errors="ignore")

    Path("data").mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print("✅ Dataset gerado com sucesso!")
    print("Aba usada:", sheet)
    print("Arquivo:", OUT_PATH)
    print("Formato:", df.shape)
    print("Target (risco_defasagem) distribuição:")
    print(df["risco_defasagem"].value_counts(normalize=True).round(4))

if __name__ == "__main__":
    main()