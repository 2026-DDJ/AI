exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]),
        categorical_features
    )
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

thresholds = np.arange(0.30, 0.701, 0.01)

rows = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X[features], y),
    start=1
):
    X_train = X[features].iloc[train_idx]
    X_test = X[features].iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=0.5,
        random_state=42
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]

    for threshold in thresholds:

        pred = (prob >= threshold).astype(int)

        rows.append({
            "fold": fold,
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy":
                balanced_accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred)
        })

result = pd.DataFrame(rows)

summary = (
    result
    .groupby("threshold")
    .agg(
        accuracy=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        f1=("f1", "mean")
    )
    .reset_index()
    .sort_values("accuracy", ascending=False)
)

print("=== Threshold별 CV 성능 TOP 15 ===")
print(summary.head(15).to_string(index=False))

best = summary.iloc[0]

print("\n=== 최고 Threshold ===")
print(f"Threshold : {best['threshold']:.2f}")
print(f"Accuracy  : {best['accuracy']:.4f}")
print(f"SD        : {best['accuracy_sd']:.4f}")
print(f"Balanced  : {best['balanced_accuracy']:.4f}")
print(f"F1        : {best['f1']:.4f}")

print("\n기존 threshold 0.50 Accuracy: 0.8213")
print(
    "변화:",
    f"{(best['accuracy'] - 0.821325) * 100:+.2f}%p"
)

summary.to_csv(
    "data/processed/threshold_results_2024.csv",
    index=False
)
