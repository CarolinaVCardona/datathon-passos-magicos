import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

DATA_PATH = "data/train.csv"
MODEL_PATH = "app/model/model.joblib"
METRICS_PATH = "data/metrics.json"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Erro: Arquivo {DATA_PATH} não encontrado. Rode o make_dataset.py primeiro.")
        return

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["risco_defasagem"])
    y = df["risco_defasagem"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    # Definição do Modelo (Regressão Logística para Explicabilidade)
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Treino e Validação
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)

    # Cálculo de Métricas
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    # Salvando o Modelo e Métricas
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Treino concluído com sucesso!")
    print(f"Métricas salvas em: {METRICS_PATH}")
    print(f"Recall: {metrics['recall']:.4f} | ROC AUC: {metrics['roc_auc']:.4f}")

    # ---  ANÁLISE DE VARIÁVEIS (EXPLICABILIDADE) ---
    print("\n" + "="*40)
    print("🔍 ANÁLISE DE IMPACTO NO RISCO")
    print("="*40)
    
    feature_names = clf.named_steps['preprocessor'].get_feature_names_out()
    weights = clf.named_steps['model'].coef_[0]
    
    importance_df = pd.DataFrame({'Atributo': feature_names, 'Peso': weights})
    importance_df['Impacto_Absoluto'] = importance_df['Peso'].abs()
    
    # As que mais aumentam o risco (Pesos positivos altos)
    print("\n📈 Variáveis que MAIS indicam risco de defasagem:")
    print(importance_df.sort_values('Peso', ascending=False).head(5)[['Atributo', 'Peso']])

    # As que mais reduzem o risco (Pesos negativos altos)
    print("\n📉 Variáveis que MAIS protegem o aluno (reduzem risco):")
    print(importance_df.sort_values('Peso', ascending=True).head(5)[['Atributo', 'Peso']])
    print("="*40)

if __name__ == "__main__":
    main()