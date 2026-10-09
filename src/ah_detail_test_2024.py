exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

# AH 계열 세부 변수 조합 확인
groups = {
    "BASE": [],
    "AH12": ["ah1", "ah2"],
    "AH1_DETAIL": [
        "ah1",
        "ah1_1_d","ah1_2_d","ah1_3_d","ah1_4_d","ah1_5_d",
        "ah1_6_d","ah1_7_d","ah1_8_d","ah1_9_d","ah1_10_d"
    ],
    "AH2_DETAIL": [
        "ah2",
        "ah2_1_d","ah2_2_d","ah2_3_d","ah2_4_d","ah2_5_d",
        "ah2_6_d","ah2_7_d","ah2_8_d","ah2_9_d","ah2_10_d"
    ],
    "AH_ALL": [
        "ah1",
        "ah1_1_d","ah1_2_d","ah1_3_d","ah1_4_d","ah1_5_d",
        "ah1_6_d","ah1_7_d","ah1_8_d","ah1_9_d","ah1_10_d",
        "ah2",
        "ah2_1_d","ah2_2_d","ah2_3_d","ah2_4_d","ah2_5_d",
        "ah2_6_d","ah2_7_d","ah2_8_d","ah2_9_d","ah2_10_d"
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, extra in groups.items():

    add = [c for c in extra if c not in data.columns]

    if add:
        temp = raw[["pidwon"] + add].copy()
        d = data.merge(
            temp,
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
        d[feats],
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
        "features": len(feats),
        "accuracy": scores["test_accuracy"].mean(),
        "sd": scores["test_accuracy"].std(),
        "f1": scores["test_f1"].mean(),
        "auc": scores["test_auc"].mean()
    })

    print(
        f"{name:10} | "
        f"Accuracy {scores['test_accuracy'].mean():.4f} | "
        f"SD {scores['test_accuracy'].std():.4f} | "
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
    "data/processed/ah_detail_results_2024.csv",
    index=False
)
