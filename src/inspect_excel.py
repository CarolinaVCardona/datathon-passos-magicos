import pandas as pd

path = "data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

xls = pd.ExcelFile(path)

print("Abas encontradas no Excel:")
for sheet in xls.sheet_names:
    print("-", sheet)

df = pd.read_excel(path, sheet_name="PEDE2024")
print(list(df.columns))

print("\nPrimeira aba:", xls.sheet_names[0])
print("Colunas:")
print(list(df.columns))

print("\nPrimeiras 3 linhas:")
print(df.head(3))