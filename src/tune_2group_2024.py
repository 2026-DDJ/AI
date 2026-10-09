import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

ind = pd.read_spss("data/raw/f_ind.sav")
df = pd.read_csv("data/processed/KHP_2024_week5_v3.csv")

util = ind[["pidwon", "ergun", "ingun", "ougun"]].copy()
util[["ergun", "ingun", "ougun"]] = (
    util[["ergun", "ingun", "ougun"]].fillna(0)
)

data = df.merge(
    util,
    on="pidwon",
    how="left",
    validate="one_to_one"
)

numeric_features = [
    "age", "income", "bmi", "chronic_count",
    "ergun", "ingun", "ougun", "hs6_1"
]

categorical_features = [
    "sex", "region1", "chronic_disease",
    "self_rated_health", "smoking_status",
    "drinking_frequency", "exercise",
    "marr", "edu", "health_ins", "disa_yn", "eco1",
    "hs1", "hs2_yn", "hs3_yn",
    "hs_eq1", "hs_eq2", "hs_eq3",
    "hs_eq4", "hs_eq5", "hs6_yn"
]

features = numeric_features + categorical_features

threshold = data["annual_medical_cost"].median()

X = data[features]
y = (
    data["annual_medical_cost"] > threshold
).astype(int)

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
        categorical_features
    )
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

experiments = [
    (
        "Random Forest",
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 10, 15],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": ["sqrt", 0.5]
        }
    ),
    (
        "Hist Gradient Boosting",
        HistGradientBoostingClassifier(
            random_state=42
        ),
        {
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__max_iter": [200, 300],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__l2_regularization": [0.5, 1.0, 2.0]
        }
    )
]

all_results = []

for name, model, params in experiments:

    print("\n====================================")
    print(name)
    print("====================================")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipe,
        param_grid=params,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    print(f"\nBest CV Accuracy: {grid.best_score_:.4f}")
    print("Best Parameters:")

    for key, value in grid.best_params_.items():
        print(f"{key}: {value}")

    all_results.append({
        "model": name,
        "cv_accuracy": grid.best_score_,
        "params": str(grid.best_params_)
    })

results = pd.DataFrame(all_results).sort_values(
    "cv_accuracy",
    ascending=False
)

print("\n\n=== 최종 튜닝 순위 ===")
print(results.to_string(index=False))

results.to_csv(
    "data/processed/classification_2group_tuning.csv",
    index=False
)

print(
    "\n저장 완료: "
    "data/processed/classification_2group_tuning.csv"
)
