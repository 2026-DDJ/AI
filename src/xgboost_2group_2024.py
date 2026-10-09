exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pandas as pd

# 전체 96 Feature 그대로 사용
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

# 과하게 큰 Grid Search 전에
# 서로 다른 성격의 XGBoost 설정을 한 번에 비교
configs = [
    {
        "name": "XGB_A",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0
    },
    {
        "name": "XGB_B",
        "n_estimators": 500,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0
    },
    {
        "name": "XGB_C",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0
    },
    {
        "name": "XGB_D",
        "n_estimators": 500,
        "learning_rate": 0.02,
        "max_depth": 4,
        "min_child_weight": 15,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0
    }
]

results = []

for config in configs:

    name = config["name"]

    params = {
        k: v
        for k, v in config.items()
        if k != "name"
    }

    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipe,
        X[features],
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced": "balanced_accuracy",
            "f1": "f1",
            "auc": "roc_auc"
        },
        n_jobs=1
    )

    acc = scores["test_accuracy"].mean()
    sd = scores["test_accuracy"].std()
    f1 = scores["test_f1"].mean()
    auc = scores["test_auc"].mean()

    results.append({
        "model": name,
        "accuracy": acc,
        "accuracy_sd": sd,
        "f1": f1,
        "roc_auc": auc
    })

    print(
        f"{name} | "
        f"Accuracy {acc:.4f} | "
        f"SD {sd:.4f} | "
        f"F1 {f1:.4f} | "
        f"AUC {auc:.4f}"
    )

result_df = pd.DataFrame(results).sort_values(
    "accuracy",
    ascending=False
)

print("\n=== XGBoost 최종 순위 ===")
print(result_df.to_string(index=False))

print("\n현재 Hist Gradient Boosting")
print("Accuracy : 0.8213")
print("F1       : 0.8201")
print("AUC      : 0.9027")

best = result_df.iloc[0]

print("\n=== 최고 XGBoost vs 현재 모델 ===")
print(f"Best XGB : {best['model']}")
print(f"Accuracy : {best['accuracy']:.4f}")
print(
    f"변화     : "
    f"{(best['accuracy'] - 0.821325) * 100:+.2f}%p"
)

result_df.to_csv(
    "data/processed/xgboost_2group_results.csv",
    index=False
)
