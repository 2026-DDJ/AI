import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src.predict import predict_medical_cost


app = FastAPI(
    title="MEDICAST AI API",
    description="개인 의료비 예측 API",
    version="1.0.0"
)


class MedicalCostInput(BaseModel):

    # 1. 가구원 수
    ingun: float

    # 2. 소득
    # 프론트에서 소득 구간에 따른 대표값 전달
    # 1구간 -> 1570.5
    # 2구간 -> 3380
    # 3구간 -> 5880
    # 4구간 -> 10160
    income: float

    # 3. 건강보험 유형
    # 미가입 선택 시 null
    health_ins: Optional[int] = None

    # 4. 거주 지역
    region1: int

    # 5. 주관적 건강상태
    self_rated_health: int

    # 6~7. 키 / 몸무게
    # API에서 BMI 자동 계산
    height: float
    weight: float

    # 8. 운동 여부
    exercise: int

    # 9. 최근 1년 외래 이용 횟수
    ougun: float

    # 10. 최근 1년 치과 이용 횟수
    ougun_dent: float

    # 11. 실손보험
    indemnity_insurance: int

    # 12. 최근 1년 입원 여부
    hospitalized_year: int

    # 13. 활동 제한
    activity_limitation: Optional[int] = None

    # 14. 불안·우울
    anxiety_depression: Optional[int] = None

    # 15. 직업군
    # 경제활동 안 함 선택 시 null
    eco9: Optional[int] = None


@app.get("/")
def root():
    return {
        "message": "MEDICAST AI API is running"
    }


@app.post("/predict")
def predict(data: MedicalCostInput):

    input_data = data.model_dump()

    # -------------------------
    # BMI 자동 계산
    # -------------------------
    height_m = input_data["height"] / 100

    if height_m > 0:
        bmi = input_data["weight"] / (height_m ** 2)
    else:
        bmi = np.nan

    input_data["bmi"] = bmi

    # 모델에는 height / weight 직접 사용하지 않음
    input_data.pop("height", None)
    input_data.pop("weight", None)

    # -------------------------
    # 치과 이용 비율 자동 계산
    # -------------------------
    if input_data["ougun"] > 0:
        dent_share = (
            input_data["ougun_dent"]
            / input_data["ougun"]
        )
    else:
        dent_share = 0

    dent_share = max(0, min(dent_share, 1))

    input_data["dent_share"] = dent_share

    # -------------------------
    # null -> NaN
    # -------------------------
    if input_data["health_ins"] is None:
        input_data["health_ins"] = np.nan

    if input_data["eco9"] is None:
        input_data["eco9"] = np.nan

    if input_data["activity_limitation"] is None:
        input_data["activity_limitation"] = np.nan

    if input_data["anxiety_depression"] is None:
        input_data["anxiety_depression"] = np.nan

    # -------------------------
    # MS 상세 의료이용 변수
    # 사용자에게 입력받지 않고 대표값 사용
    # -------------------------
    input_data.update({
        "ms_total_days": 17,
        "ms_in_treatment_nunique": 0,
        "ms_facility_nunique": 2,
        "ms_mtype_1_count": 0,
        "ms_mtype_3_count": 0,
        "ms_in_reason_2_count": 0,
        "ms_in_trt_1_count": 0,
        "ms_in_choice_1_count": 0,
        "ms_in_care2_2_count": 0,
        "ms_in_dsc_nunique": 0,
        "ms_dent_trt_10_count": 0,
        "ms_dent_trt_5_count": 0,
        "ms_in_sat7_mean": 0
    })

    # -------------------------
    # 예측
    # -------------------------
    result = predict_medical_cost(input_data)

    return {
        "success": True,
        "predictedAnnualMedicalCost": round(
            result["predicted_annual_medical_cost"]
        ),
        "highCostProbability": round(
            result["high_cost_probability"],
            4
        )
    }