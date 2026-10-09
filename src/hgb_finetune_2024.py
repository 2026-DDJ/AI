exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

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

model = HistGradientBoostingClassifier(
    random_state=42
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

param_dist = {
    "model__learning_rate": [
        0.015, 0.02, 0.025, 0.03,
        0.035, 0.04, 0.05
    ],
    "model__max_iter": [
        150, 200, 250, 300, 400
    ],
    "model__max_leaf_nodes": [
        7, 10, 12, 15, 18, 20, 25, 31
    ],
    "model__max_depth": [
        None, 3, 4, 5, 6
    ],
    "model__min_samples_leaf": [
        10, 15, 20, 25, 30, 40
    ],
    "model__l2_regularization": [
        0, 0.1, 0.3, 0.5, 0.8,
        1.0, 1.5, 2.0, 3.0
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=80,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=False
)

search.fit(X[features], y)

print("\n=== BEST ===")
print(f"CV Accuracy: {search.best_score_:.4f}")

for k, v in search.best_params_.items():
    print(k, "=", v)

results = pd.DataFrame(search.cv_results_)

top = results[
    [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "params"
    ]
].sort_values(
    "rank_test_score"
).head(15)

print("\n=== TOP 15 ===")
print(top.to_string(index=False))

print("\n현재 최고 Accuracy: 0.8213")
print(
    "변화:",
    f"{(search.best_score_ - 0.821325) * 100:+.2f}%p"
)

top.to_csv(
    "data/processed/hgb_finetune_results.csv",
    index=False
)
