import joblib
import pandas as pd
import numpy as np


# ============================================================
# LOAD FINAL MODELS
# ============================================================

classifier = joblib.load("models/classifier.pkl")
low_model = joblib.load("models/low_model.pkl")
high_model = joblib.load("models/high_model.pkl")
metadata = joblib.load("models/metadata.pkl")

FEATURES = metadata["features"]
SCALE = metadata["scale"]


# ============================================================
# PREDICT
# ============================================================

def predict_medical_cost(input_data):

    # dictionary -> DataFrame
    X = pd.DataFrame([input_data])

    # 모델이 학습한 28개 feature 순서로 정렬
    X = X.reindex(columns=FEATURES)

    # 고비용군일 확률
    high_probability = classifier.predict_proba(X)[:, 1][0]

    # 저비용군 모델 예측
    low_prediction = low_model.predict(X)[0]

    # 고비용군 모델 예측
    high_prediction = high_model.predict(X)[0]

    # Soft Gating
    final_prediction = (
        (1 - high_probability) * low_prediction
        + high_probability * (high_prediction * SCALE)
    )

    # 의료비가 음수가 되는 경우 방지
    final_prediction = max(0, final_prediction)

    return {
        "predicted_annual_medical_cost": int(round(final_prediction)),
        "high_cost_probability": round(float(high_probability), 4),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_input = {
        "ingun": 2,
        "income": 3000,
        "health_ins": 1,
        "region1": 1,
        "self_rated_health": 2,
        "bmi": 22.0,
        "exercise": 1,
        "ougun": 5,
        "ougun_dent": 1,
        "dent_share": 0.2,
        "indemnity_insurance": 1,
        "hospitalized_year": 0,
        "activity_limitation": 1,
        "anxiety_depression": 1,
        "eco9": 1,

        # MS 상세 변수
        # 현재 데모에서는 입력하지 않은 값은 기본값 0 사용
        "ms_total_days": 0,
        "ms_in_treatment_nunique": 0,
        "ms_facility_nunique": 0,
        "ms_mtype_1_count": 0,
        "ms_mtype_3_count": 0,
        "ms_in_reason_2_count": 0,
        "ms_in_trt_1_count": 0,
        "ms_in_choice_1_count": 0,
        "ms_in_care2_2_count": 0,
        "ms_in_dsc_nunique": 0,
        "ms_dent_trt_10_count": 0,
        "ms_dent_trt_5_count": 0,
        "ms_in_sat7_mean": 0,
    }

    result = predict_medical_cost(test_input)

    print("=" * 60)
    print("MEDICAST PREDICTION TEST")
    print("=" * 60)

    print(
        f"예측 연간 의료비 : "
        f"{result['predicted_annual_medical_cost']:,}원"
    )

    print(
        f"고비용군 확률    : "
        f"{result['high_cost_probability']:.2%}"
    )

    print("=" * 60)