# 기존 확장 모델의 데이터 구성 부분 재사용
exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd

importance = pd.read_csv(
    "data/processed/feature_importance_2group.csv"
)

# 중요도 순서
ranked = importance["feature"].tolist()

# 여러 Feature 개수를 한 번에 비교
subset_sizes = [5, 10, 15, 20, 24, 30, 40, 60, 96]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for n in subset_sizes:

    selected = ranked[:n]

    selected_num = [
        x for x in selected
        if x in numeric_features
    ]

    selected_cat = [
        x for x in selected
        if x in categorical_features
    ]

    transformers = []

    if selected_num:
        transformers.append(
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    )
                ]),
                selected_num
            )
        )

    if selected_cat:
        transformers.append(
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]),
                selected_cat
            )
        )

    prep = ColumnTransformer(transformers)

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
        X[selected],
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1": "f1",
            "auc": "roc_auc"
        },
        n_jobs=-1
    )

    acc = scores["test_accuracy"].mean()
    sd = scores["test_accuracy"].std()
    f1 = scores["test_f1"].mean()
    auc = scores["test_auc"].mean()

    results.append({
        "features": n,
        "accuracy": acc,
        "accuracy_sd": sd,
        "f1": f1,
        "roc_auc": auc
    })

    print(
        f"{n:>2} Features | "
        f"Accuracy {acc:.4f} | "
        f"SD {sd:.4f} | "
        f"F1 {f1:.4f} | "
        f"AUC {auc:.4f}"
    )

result_df = pd.DataFrame(results).sort_values(
    "accuracy",
    ascending=False
)

print("\n=== Feature 개수별 최종 순위 ===")
print(result_df.to_string(index=False))

best_n = int(result_df.iloc[0]["features"])

print("\nBest Feature 수:", best_n)
print("\n=== 선택 Feature ===")

for x in ranked[:best_n]:
    print(x)

result_df.to_csv(
    "data/processed/feature_subset_results.csv",
    index=False
)
