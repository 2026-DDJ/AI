import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# 1. 데이터
# ==================================================

ind = pd.read_spss("data/raw/f_ind.sav")
df = pd.read_csv("data/processed/KHP_2024_week5_v3.csv")

util = ind[
    ["pidwon", "ergun", "ingun", "ougun"]
].copy()

util[["ergun", "ingun", "ougun"]] = (
    util[["ergun", "ingun", "ougun"]].fillna(0)
)

data = df.merge(
    util,
    on="pidwon",
    how="left",
    validate="one_to_one"
)

# ==================================================
# 2. Feature
# ==================================================

numeric_features = [
    "age",
    "income",
    "bmi",
    "chronic_count",
    "ergun",
    "ingun",
    "ougun",
    "hs6_1"
]

categorical_features = [
    "sex",
    "region1",
    "chronic_disease",
    "self_rated_health",
    "smoking_status",
    "drinking_frequency",
    "exercise",
    "marr",
    "edu",
    "health_ins",
    "disa_yn",
    "eco1",
    "hs1",
    "hs2_yn",
    "hs3_yn",
    "hs_eq1",
    "hs_eq2",
    "hs_eq3",
    "hs_eq4",
    "hs_eq5",
    "hs6_yn"
]

features = numeric_features + categorical_features

# ==================================================
# 3. 의료비 타깃 3구간 생성
# ==================================================

cost = data["annual_medical_cost"]

q33 = cost.quantile(1/3)
q67 = cost.quantile(2/3)

data["cost_group"] = pd.cut(
    cost,
    bins=[-np.inf, q33, q67, np.inf],
    labels=["LOW", "MID", "HIGH"],
    include_lowest=True
)

print("=== 의료비 구간 ===")
print(f"LOW  : ~ {q33:,.0f}원")
print(f"MID  : {q33:,.0f} ~ {q67:,.0f}원")
print(f"HIGH : {q67:,.0f}원 초과")

print("\n=== 클래스 분포 ===")
print(data["cost_group"].value_counts().sort_index())
print(
    data["cost_group"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

# ==================================================
# 4. X / y
# ==================================================

X = data[features]
y = data["cost_group"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==================================================
# 5. 전처리
# ==================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
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
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])

# ==================================================
# 6. 모델
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
}

# ==================================================
# 7. Holdout 비교
# ==================================================

results = []

print("\n\n========================================")
print("HOLDOUT MODEL COMPARISON")
print("========================================")

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)
    f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )

    results.append({
        "model": name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": f1
    })

    print(f"\n=== {name} ===")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"Macro F1          : {f1:.4f}")

# ==================================================
# 8. 결과 순위
# ==================================================

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "macro_f1",
    ascending=False
)

print("\n\n=== Holdout 순위 ===")
print(result_df.to_string(index=False))

# ==================================================
# 9. 가장 좋은 모델 5-Fold CV
# ==================================================

best_name = result_df.iloc[0]["model"]
best_classifier = models[best_name]

best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_classifier)
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    best_pipeline,
    X,
    y,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro"
    },
    n_jobs=-1
)

print("\n\n========================================")
print("BEST MODEL 5-FOLD CV")
print("========================================")

print("Best Model:", best_name)

for i in range(5):
    print(
        f"Fold {i+1} | "
        f"Accuracy {scores['test_accuracy'][i]:.4f} | "
        f"Balanced {scores['test_balanced_accuracy'][i]:.4f} | "
        f"Macro F1 {scores['test_f1_macro'][i]:.4f}"
    )

print("\n=== CV 평균 ===")
print(
    f"Accuracy          : "
    f"{scores['test_accuracy'].mean():.4f}"
)
print(
    f"Accuracy SD       : "
    f"{scores['test_accuracy'].std():.4f}"
)
print(
    f"Balanced Accuracy : "
    f"{scores['test_balanced_accuracy'].mean():.4f}"
)
print(
    f"Macro F1          : "
    f"{scores['test_f1_macro'].mean():.4f}"
)

# ==================================================
# 10. 최종 Holdout 상세 결과
# ==================================================

best_pipeline.fit(X_train, y_train)

best_pred = best_pipeline.predict(X_test)

print("\n\n=== 최종 Classification Report ===")
print(
    classification_report(
        y_test,
        best_pred,
        digits=4
    )
)

print("=== Confusion Matrix ===")
print(
    confusion_matrix(
        y_test,
        best_pred,
        labels=["LOW", "MID", "HIGH"]
    )
)

# ==================================================
# 11. 결과 저장
# ==================================================

result_df.to_csv(
    "data/processed/classification_model_results.csv",
    index=False
)

print(
    "\n저장 완료: "
    "data/processed/classification_model_results.csv"
)
