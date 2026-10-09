exec(open("src/feature_expand_2group_2024.py").read().split("# =========================\n# 5-Fold CV")[0])

import os
import joblib

os.makedirs("models", exist_ok=True)

# 전체 2024 데이터로 최종 학습
pipeline.fit(
    X[features],
    y
)

joblib.dump(
    pipeline,
    "models/medical_cost_classifier_2024.joblib"
)

joblib.dump(
    features,
    "models/medical_cost_features_2024.joblib"
)

metadata = {
    "target": "annual_medical_cost",
    "classification_threshold": 687550,
    "classes": {
        0: "LOW",
        1: "HIGH"
    },
    "n_samples": len(data),
    "n_features": len(features),
    "model": "HistGradientBoostingClassifier",
    "learning_rate": 0.03,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "l2_regularization": 0.5,
    "repeated_cv_accuracy": 0.81924,
    "repeated_cv_sd": 0.00669,
    "repeated_cv_f1": 0.81804,
    "repeated_cv_auc": 0.90214
}

joblib.dump(
    metadata,
    "models/medical_cost_metadata_2024.joblib"
)

print("=== 최종 모델 저장 완료 ===")
print("모델: models/medical_cost_classifier_2024.joblib")
print("Feature: models/medical_cost_features_2024.joblib")
print("Metadata: models/medical_cost_metadata_2024.joblib")
print("Feature 수:", len(features))
