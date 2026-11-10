MEDICAST AI

개인의 건강정보와 의료이용 정보를 기반으로 연간 의료비를 예측하는 MEDICAST의 AI 모듈입니다.

한국의료패널(KHP) 2024년 연간 데이터를 활용하며, 일반 의료비와 고액 의료비의 분포 차이를 고려한 Two-stage XGBoost 모델을 사용합니다.

주요 기능

* 개인 연간 의료비 예측
* 고액 의료비 발생 확률 예측
* 건강·의료이용 정보를 활용한 개인별 예측
* 사용자 입력 조건 변경에 따른 의료비 재예측
* FastAPI 기반 백엔드 연동 API 제공

데이터

한국의료패널(KHP)

2024년 연간 데이터를 모델 학습에 활용했습니다.

연간 의료비는 응급, 입원, 외래 의료비를 합산하여 구성했으며, 의료비를 직접 구성하는 금액 변수는 모델 입력 Feature에서 제외했습니다.

원본 한국의료패널 데이터 및 가공 데이터는 저장소에 포함하지 않습니다.

사용자 입력

서비스에서는 다음 15개 항목을 입력받습니다.

1. 가구원 수
2. 소득
3. 건강보험 유형
4. 거주 지역
5. 주관적 건강상태
6. 키
7. 몸무게
8. 운동 여부
9. 최근 1년 외래 이용 횟수
10. 최근 1년 치과 이용 횟수
11. 실손보험 가입 여부
12. 최근 1년 입원 여부
13. 활동 제한 정도
14. 불안·우울 정도
15. 직업군

키와 몸무게를 이용해 BMI를 자동 계산하며, 외래 및 치과 이용 정보를 이용해 치과 이용 비율을 파생변수로 생성합니다.

소득은 학습데이터의 분포를 기준으로 4개 구간으로 구성하고 각 구간의 중앙값을 모델 입력값으로 사용합니다.

모델 Feature

최종 모델은 총 28개 Feature를 사용합니다.

기본 및 사용자 관련 Feature

* 가구원 수
* 소득
* 건강보험 유형
* 지역
* 주관적 건강상태
* BMI
* 운동 여부
* 외래 이용 횟수
* 치과 외래 이용 횟수
* 치과 이용 비율
* 실손보험 가입 여부
* 입원 여부
* 활동 제한
* 불안·우울
* 직업군

의료이용 상세 Feature

한국의료패널 의료이용 상세 데이터에서 생성한 13개 Feature를 추가로 사용합니다.

웹 프로토타입에서는 사용자가 직접 입력하기 어려운 의료이용 상세 Feature에 학습데이터 기반 대표값을 적용합니다.

모델 구조

의료비 데이터의 긴 꼬리 분포와 고액 의료비 예측 문제를 고려하여 Two-stage XGBoost 구조를 사용합니다.

1. 고액 의료비 발생 확률 분류
2. 일반 의료비군 회귀 예측
3. 고액 의료비군 회귀 예측
4. 분류 확률을 이용한 Soft Gating
5. 최종 연간 의료비 산출

고액 의료비 기준은 연간 의료비 상위 15%로 설정했습니다.

모델 성능

28개 Feature를 사용한 반복 검증 결과:

* 평균 R²: 0.5841
* 평균 MAE: 690,318원
* 평균 RMSE: 1,565,846원
* 평균 AUC: 0.9341

위 성능은 실제 28개 Feature를 사용한 모델 검증 결과입니다.

웹 프로토타입에서는 의료이용 상세 Feature 일부에 대표값을 적용하므로 실제 서비스 입력 환경과 모델 검증 환경에는 차이가 있습니다.

AI API

FastAPI를 이용해 예측 API를 제공합니다.

Endpoint

POST /predict

요청 예시

{
  "ingun": 2,
  "income": 5880,
  "health_ins": 1,
  "region1": 11,
  "self_rated_health": 3,
  "height": 165,
  "weight": 55,
  "exercise": 1,
  "ougun": 5,
  "ougun_dent": 1,
  "indemnity_insurance": 1,
  "hospitalized_year": 0,
  "activity_limitation": 1,
  "anxiety_depression": 1,
  "eco9": 31
}

응답 예시

{
  "success": true,
  "predictedAnnualMedicalCost": 1275157,
  "highCostProbability": 0.0276
}

실행 방법

패키지 설치

pip install -r requirements.txt

API 실행

uvicorn api.main:app --reload

API 문서:

http://127.0.0.1:8000/docs

프로젝트 구조

AI/
├── api/
│   └── main.py
├── data/
│   ├── processed/
│   └── raw/
├── models/
│   ├── classifier.pkl
│   ├── high_model.pkl
│   ├── low_model.pkl
│   └── metadata.pkl
├── src/
│   ├── predict.py
│   └── train_final_model.py
├── .gitignore
├── requirements.txt
└── README.md

모델 파일

* classifier.pkl : 고액 의료비 분류 모델
* low_model.pkl : 일반 의료비군 회귀 모델
* high_model.pkl : 고액 의료비군 회귀 모델
* metadata.pkl : Feature 목록 및 모델 설정 정보

서비스 연동 구조

Frontend
   ↓
Spring Boot Backend
   ↓
FastAPI AI Server
   ↓
Two-stage XGBoost Model
   ↓
연간 의료비 + 고액 의료비 확률
   ↓
Backend
   ↓
Frontend
