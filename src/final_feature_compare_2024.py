exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

groups = {
    "BASE_96": [],
    "AH12_98": ["ah1", "ah2"]
}

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

for name, add in groups.items():

    if add:
        d = data.merge(
            raw[["pidwon"] + add],
            on="pidwon",
            how="left",
            validate="one_to_one"
        )
    else:
        d = data.copy()

    num = numeric_features.copy()
    cat = categorical_features + add
    feats = num + cat

    prep = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            num
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ]),
            cat
        )
    ])

    pipe = Pipeline([
        ("preprocessor", prep),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=0.5,
            random_state=42
        ))
    ])

    scores = cross_validate(
        pipe,
        d[feats],
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced": "balanced_accuracy",
            "f1": "f1",
            "auc": "roc_auc"
        },
        n_jobs=-1
    )

    print(f"\n=== {name} ===")
    print("검증 횟수:", len(scores["test_accuracy"]))
    print(f"Accuracy : {scores['test_accuracy'].mean():.5f}")
    print(f"SD       : {scores['test_accuracy'].std():.5f}")
    print(f"Balanced : {scores['test_balanced'].mean():.5f}")
    print(f"F1       : {scores['test_f1'].mean():.5f}")
    print(f"ROC-AUC  : {scores['test_auc'].mean():.5f}")
