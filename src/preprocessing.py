import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# =========================
# 1. 데이터 불러오기
# =========================

DATA_PATH = "data/processed/KHP_2024_week3_final_corrected.csv"

df = pd.read_csv(DATA_PATH)

print("원본 데이터:", df.shape)


# =========================
# 2. 모델에 사용할 변수
# =========================

feature_cols = [
    "age",
    "sex",
    "region1",
    "income",
    "bmi",
    "chronic_disease",
    "self_rated_health",
    "smoking_status",
    "drinking_frequency",
    "exercise"
]

target_col = "annual_medical_cost"

df = df[feature_cols + [target_col]]


# =========================
# 3. 결측치 처리
# =========================

df = df.dropna()

print("결측치 처리 후:", df.shape)


# =========================
# 4. X / y 분리
# =========================

X = df[feature_cols]
y = df[target_col]


# =========================
# 5. 변수 유형 구분
# =========================

numeric_cols = [
    "age",
    "income",
    "bmi"
]

categorical_cols = [
    "sex",
    "region1",
    "chronic_disease",
    "self_rated_health",
    "smoking_status",
    "drinking_frequency",
    "exercise"
]


# =========================
# 6. 범주형 변수 인코딩
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        ),
        (
            "num",
            "passthrough",
            numeric_cols
        )
    ]
)

X_encoded = preprocessor.fit_transform(X)

print("인코딩 후 X:", X_encoded.shape)


# =========================
# 7. Train / Test 분리
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

print("\n=== 최종 결과 ===")
print("Train:", X_train.shape)
print("Test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)