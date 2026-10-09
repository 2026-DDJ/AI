exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

# 다음 추가 후보 그룹
groups = {
    "BASE": [],

    # 입원 여부
    "IN1YEAR": ["in1year"],

    # 처방약 조제 여부
    "AH3": ["ah3"],

    # 상용치료의사 관련
    "USC": ["usc5", "usc6", "usc7", "usc8"],

    # 정신건강 관련
    "MENTAL": [
        "hs_type1_d",
        "hs_type2_d",
        "hs_type3_d",
        "hs_type4_d",
        "hs_type5_d",
        "hs_med_yn"
    ],

    # 의료이용 행태
    "PHI_BEHAVIOR": ["i_phi5", "i_phi6"],

    # 개별적으로 효과 있던 AH 포함
    "AH12": ["ah1", "ah2"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, requested in groups.items():

    # 실제 raw 데이터에 존재하는 변수만 사용
    add = [
        c for c in requested
        if c in raw.columns and c not in data.columns
    ]

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

    results.append({
        "group": name,
        "added": ",".join(add),
        "n_features": len(feats),
        "accuracy": scores["test_accuracy"].mean(),
        "sd": scores["test_accuracy"].std(),
        "balanced": scores["test_balanced"].mean(),
        "f1": scores["test_f1"].mean(),
        "auc": scores["test_auc"].mean()
    })

    print(
        f"{name:14} | "
        f"추가 {add} | "
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
    "data/processed/additional_feature_results_2024.csv",
    index=False
)
