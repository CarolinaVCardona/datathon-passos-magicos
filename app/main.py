from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import numpy as np
from typing import Dict, Any

MODEL_PATH = "app/model/model.joblib"
DATA_SAMPLE_PATH = "data/train.csv"

app = FastAPI(
    title="Passos Mágicos - API de Predição de Risco",
    description="Sistema de detecção precoce de defasagem escolar.",
    version="1.2.0"
)

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

def get_sample_data():
    """Busca um exemplo real para preencher o Swagger automaticamente."""
    try:
        if os.path.exists(DATA_SAMPLE_PATH):
            df = pd.read_csv(DATA_SAMPLE_PATH)
            if "risco_defasagem" in df.columns:
                df = df.drop(columns=["risco_defasagem"])
            return df.iloc[0].replace({np.nan: None}).to_dict()
    except Exception:
        pass
    return {"mensagem": "Execute o treino para carregar exemplos reais."}

class PredictRequest(BaseModel):
    """Corpo da requisição com exemplo automático."""
    data: Dict[str, Any] = Field(..., example=get_sample_data())

@app.get("/")
def health():
    return {"status": "online", "modelo_carregado": model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Realiza a predição tratando colunas ausentes e tipos de dados nulos.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não encontrado.")
    
    try:
        X = pd.DataFrame([req.data])
        
        expected_cols = model.named_steps['preprocessor'].feature_names_in_
        
        for col in expected_cols:
            if col not in X.columns:
                X[col] = None
        
        X = X[expected_cols]
        
        X = X.where(pd.notnull(X), None)
        
        proba = float(model.predict_proba(X)[:, 1][0])
        pred = int(proba >= 0.5)
        
        return {
            "risco_defasagem": pred, 
            "probabilidade": round(proba, 4),
            "status": "Atenção: Aluno em Risco" if pred == 1 else "Estudante Estável"
        }
    except Exception as e:
        print(f"Erro interno: {e}")
        raise HTTPException(status_code=400, detail=f"Erro no processamento: {str(e)}")