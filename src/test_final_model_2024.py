# 학습 때 사용한 96개 Feature 데이터 생성 부분만 실행
exec(
    open("src/feature_expand_2group_2024.py")
    .read()
    .split("# =========================\n# 5-Fold CV")[0]
)

import joblib
import pandas as pd

MODEL_PATH = "models/medical_cost_classifier_2024.joblib"
FEATURE_PATH = "models/medical_cost_features_2024.joblib"

model = joblib.load(MODEL_PATH)
features_saved = joblib.load(FEATURE_PATH)

print("=== 최종 모델 테스트 ===")

# 저장된 feature와 현재 생성된 feature 일치 확인
print("저장 Feature 수:", len(features_saved))
print("현재 Feature 수:", len(features))

missing = [c for c in features_saved if c not in X.columns]

if missing:
    print("누락 Feature:", missing)
    raise ValueError("최종 모델 입력 Feature 생성 실패")

# 테스트용 10명
test = X[features_saved].head(10)

pred = model.predict(test)
prob = model.predict_proba(test)[:, 1]

actual_cost = data["annual_medical_cost"].head(10)

result = pd.DataFrame({
    "실제_의료비": actual_cost.values,
    "실제등급": [
        "HIGH" if x > 687550 else "LOW"
        for x in actual_cost
    ],
    "예측등급": [
        "HIGH" if x == 1 else "LOW"
        for x in pred
    ],
    "HIGH_확률": prob
})

print("\n=== 예측 결과 ===")
print(result.to_string(index=False))

print("\n모델 로드: 정상")
print("Feature 일치: 정상")
print("Feature 수:", len(features_saved))
print("예측 건수:", len(pred))
