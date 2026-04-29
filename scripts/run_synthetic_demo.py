from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic" / "synthetic_model_frame.csv"
OUT = ROOT / "outputs" / "demo" / "synthetic_demo_metrics.csv"


def fit_task(df: pd.DataFrame, target: str) -> dict[str, float | str]:
    feature_cols = [
        col
        for col in df.columns
        if col not in {"person_id", "year", "is_biased", "phlegm_damp_any"}
    ]
    x = df[feature_cols]
    y = df[target].astype(int)

    numeric_cols = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in x.columns if col not in numeric_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260429),
            ),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=20260429, stratify=y
    )
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    return {
        "target": target,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "event_rate_test": float(y_test.mean()),
        "auc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
    }


def main() -> None:
    df = pd.read_csv(DATA)
    rows = [fit_task(df, "phlegm_damp_any"), fit_task(df, "is_biased")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
