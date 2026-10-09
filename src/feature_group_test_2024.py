exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

groups = {
    "BASE_96": [],
    "CARE": ["care1"],
    "AH": ["ah1", "ah2"],
    "FFS": ["i_ffs_yn"],
    "ECO": [
        "eco2","eco3","eco4","eco5",
        "eco6","eco8","eco9","eco10"
    ],
    "CARE_AH": ["care1","ah1","ah2"],
    "AH_FFS": ["ah1","ah2","i_ffs_yn"],
    "CARE_FFS": ["care1","i_ffs_yn"],
    "ALL_12": [
        "care1","ah1","ah2","i_ffs_yn",
        "eco2","eco3","eco4","eco5",
        "eco6","eco8","eco9","eco10"
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, extra_cols in groups.items():

    add_cols = [
        c for c in extra_cols
        if c not in data.columns
    ]

    if add_cols:
        temp = raw[["pidwon"] + add_cols].copy()
        test_data = data.merge(
            temp,
            on="pidwon",
            how="left",
            validate="one_to_one"
        )
    else:
        test_data = data.copy()

    num = numeric_features.copy()
    cat = categorical_features + add_cols
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
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]),
            cat
        )
    ])

    model = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=0.5,
        random_state=42
    )

    pipe = Pipeline([
        ("preprocessor", prep),
        ("model", model)
    ])

    scores = cross_validate(
        pipe,
        test_data[feats],
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1": "f1",
            "auc": "roc_auc"
        },
        n_jobs=-1
    )

    results.append({
        "group": name,
        "n_features": len(feats),
        "accuracy": scores["test_accuracy"].mean(),
        "sd": scores["test_accuracy"].std(),
        "f1": scores["test_f1"].mean(),
        "auc": scores["test_auc"].mean()
    })

    print(
        f"{name:10} | "
        f"Accuracy {scores['test_accuracy'].mean():.4f} | "
        f"F1 {scores['test_f1'].mean():.4f} | "
        f"AUC {scores['test_auc'].mean():.4f}"
    )

result = pd.DataFrame(results).sort_values(
    "accuracy",
    ascending=False
)

print("\n=== 최종 순위 ===")
print(result.to_string(index=False))

result.to_csv(
    "data/processed/feature_group_results_2024.csv",
    index=False
)
