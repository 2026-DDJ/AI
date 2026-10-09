# 기존 확장 모델 파일을 불러와 동일한 데이터/feature 구성 사용
exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

baseline = accuracy_score(
    y_test,
    pipeline.predict(X_test)
)

print("\n=== Baseline Holdout Accuracy ===")
print(f"{baseline:.4f}")

print("\nPermutation Importance 계산 중...")

result = permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": result.importances_mean,
    "std": result.importances_std
}).sort_values(
    "importance",
    ascending=False
)

print("\n=== Feature Importance ===")
print(importance.to_string(index=False))

importance.to_csv(
    "data/processed/feature_importance_2group.csv",
    index=False
)

print(
    "\n저장 완료: "
    "data/processed/feature_importance_2group.csv"
)

print("\n=== 중요도 > 0 Feature ===")
positive = importance[
    importance["importance"] > 0
]

print(positive.to_string(index=False))
print("\n개수:", len(positive))
