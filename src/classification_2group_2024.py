import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
    confusion_matrix,
    roc_auc_score
)

# 데이터
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

# =========================================
# 2구간 타깃
# 중앙값 기준으로 정확히 균형 분류
# =========================================

threshold = data["annual_medical_cost"].median()

data["cost_group"] = np.where(
    data["annual_medical_cost"] > threshold,
    1,
    0
)

print("=== 2구간 의료비 분류 ===")
print(f"기준 의료비: {threshold:,.0f}원")
print(data["cost_group"].value_counts().sort_index())
print(
    data["cost_group"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

X = data[features]
y = data["cost_group"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
}

results = []

print("\n=== HOLDOUT ===")

for name, model in models.items():

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    bal = balanced_accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    results.append({
        "model": name,
        "accuracy": acc,
        "balanced_accuracy": bal,
        "f1": f1,
        "roc_auc": auc
    })

    print(f"\n{name}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bal:.4f}")
    print(f"F1                : {f1:.4f}")
    print(f"ROC-AUC           : {auc:.4f}")

results = pd.DataFrame(results).sort_values(
    "balanced_accuracy",
    ascending=False
)

print("\n=== 순위 ===")
print(results.to_string(index=False))

best_name = results.iloc[0]["model"]
best_model = models[best_name]

best_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    best_pipe,
    X,
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

print("\n=== BEST MODEL 5-FOLD CV ===")
print("Best Model:", best_name)

for i in range(5):
    print(
        f"Fold {i+1} | "
        f"Accuracy {scores['test_accuracy'][i]:.4f} | "
        f"Balanced {scores['test_balanced'][i]:.4f} | "
        f"F1 {scores['test_f1'][i]:.4f} | "
        f"AUC {scores['test_auc'][i]:.4f}"
    )

print("\n=== CV 평균 ===")
print(f"Accuracy          : {scores['test_accuracy'].mean():.4f}")
print(f"Accuracy SD       : {scores['test_accuracy'].std():.4f}")
print(f"Balanced Accuracy : {scores['test_balanced'].mean():.4f}")
print(f"F1                : {scores['test_f1'].mean():.4f}")
print(f"ROC-AUC           : {scores['test_auc'].mean():.4f}")

best_pipe.fit(X_train, y_train)
pred = best_pipe.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(
    y_test,
    pred,
    target_names=["LOW", "HIGH"],
    digits=4
))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, pred))

results.to_csv(
    "data/processed/classification_2group_results.csv",
    index=False
)

print(
    "\n저장 완료: "
    "data/processed/classification_2group_results.csv"
)
