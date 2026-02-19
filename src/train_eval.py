import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fairness import fairness_by_group

DATA_PATH = Path("data/compas-scores-two-years.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH)
    return df

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Mantemos colunas comuns do arquivo da ProPublica
    # Target clássico: two_year_recid (0/1)
    needed = [
        "age", "sex", "race", "priors_count", "c_charge_degree", "two_year_recid"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    dfx = df[needed].copy()

    # Limpeza mínima
    dfx = dfx.dropna()
    # Garantir tipos
    dfx["two_year_recid"] = dfx["two_year_recid"].astype(int)
    dfx["priors_count"] = pd.to_numeric(dfx["priors_count"], errors="coerce")
    dfx = dfx.dropna()
    dfx["priors_count"] = dfx["priors_count"].astype(int)

    return dfx

def train_and_evaluate(dfx: pd.DataFrame, random_state: int = 42):
    X = dfx.drop(columns=["two_year_recid"])
    y = dfx["two_year_recid"]

    cat_cols = ["sex", "race", "c_charge_degree"]
    num_cols = ["age", "priors_count"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = LogisticRegression(max_iter=200, solver="lbfgs")

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "n_total": int(len(dfx)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_test": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "random_state": random_state,
        "test_size": 0.3,
        "model": "LogisticRegression",
    }

    # Dataframe com y_true/y_pred + grupos para fairness
    eval_df = X_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = y_pred

    fair_race = fairness_by_group(eval_df, "race", "y_true", "y_pred")
    fair_sex = fairness_by_group(eval_df, "sex", "y_true", "y_pred")

    return metrics, fair_race, fair_sex

def main():
    df = load_data()
    dfx = prepare_dataset(df)
    metrics, fair_race, fair_sex = train_and_evaluate(dfx)

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fair_race.to_csv(OUT_DIR / "fairness_by_race.csv", index=False)
    fair_sex.to_csv(OUT_DIR / "fairness_by_sex.csv", index=False)

    print("OK! Arquivos gerados em /outputs:")
    print("- metrics.json")
    print("- fairness_by_race.csv")
    print("- fairness_by_sex.csv")
    print("\nResumo:", metrics)

if __name__ == "__main__":
    main()
