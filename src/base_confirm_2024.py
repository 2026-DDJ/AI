exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

scores = cross_validate(
    pipeline,
    X[features],
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

print("=== 기존 96 Feature 반복 교차검증 ===")
print("검증 횟수:", len(scores["test_accuracy"]))
print(f"Accuracy : {scores['test_accuracy'].mean():.4f}")
print(f"SD       : {scores['test_accuracy'].std():.4f}")
print(f"Min      : {scores['test_accuracy'].min():.4f}")
print(f"Max      : {scores['test_accuracy'].max():.4f}")
print(f"Balanced : {scores['test_balanced'].mean():.4f}")
print(f"F1       : {scores['test_f1'].mean():.4f}")
print(f"ROC-AUC  : {scores['test_auc'].mean():.4f}")

print("\n각 Fold Accuracy")
print(np.round(scores["test_accuracy"], 4))
