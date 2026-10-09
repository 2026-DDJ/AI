exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

# 추가 후보
new_features = [
    # 돌봄
    "care1",

    # 미충족 의료
    "ah1",
    "ah2",

    # 실손보험
    "i_ffs_yn",

    # 경제활동/직업
    "eco2",
    "eco3",
    "eco4",
    "eco5",
    "eco6",
    "eco8",
    "eco9",
    "eco10"
]

raw_extra = raw[
    ["pidwon"] + new_features
].copy()

# 기존 데이터에 추가
add_cols = [
    c for c in new_features
    if c not in data.columns
]

data2 = data.merge(
    raw_extra[["pidwon"] + add_cols],
    on="pidwon",
    how="left",
    validate="one_to_one"
)

# 전부 범주형으로 처리
categorical_v2 = categorical_features + add_cols
numeric_v2 = numeric_features

features_v2 = numeric_v2 + categorical_v2

X2 = data2[features_v2]
y2 = (
    data2["annual_medical_cost"]
    > data2["annual_medical_cost"].median()
).astype(int)

print("=== V2 Feature ===")
print("전체 Feature:", len(features_v2))
print("추가 Feature:", add_cols)

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numeric_v2
    ),
    (
        "cat",
        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]),
        categorical_v2
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
    ("preprocessor", preprocessor),
    ("model", model)
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    pipe,
    X2,
    y2,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "balanced": "balanced_accuracy",
        "f1": "f1",
        "auc": "roc_auc"
    },
    n_jobs=-1
)

print("\n=== 5-Fold ===")

for i in range(5):
    print(
        f"Fold {i+1} | "
        f"Accuracy {scores['test_accuracy'][i]:.4f} | "
        f"F1 {scores['test_f1'][i]:.4f} | "
        f"AUC {scores['test_auc'][i]:.4f}"
    )

acc = scores["test_accuracy"].mean()

print("\n=== 평균 ===")
print(f"Accuracy : {acc:.4f}")
print(f"SD       : {scores['test_accuracy'].std():.4f}")
print(f"Balanced : {scores['test_balanced'].mean():.4f}")
print(f"F1       : {scores['test_f1'].mean():.4f}")
print(f"ROC-AUC  : {scores['test_auc'].mean():.4f}")

print("\n기존 Accuracy: 0.8213")
print(
    "변화:",
    f"{(acc - 0.821325) * 100:+.2f}%p"
)
