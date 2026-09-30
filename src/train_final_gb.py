import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. 데이터 불러오기
# =========================

prev = pd.read_spss("data/raw/e_ind.sav")

current = pd.read_csv(
    "data/processed/KHP_2024_week5_v3.csv"
)


# =========================
# 2. 2023 Feature 선택
# =========================

features = [
    "pidwon",
    "ergun",
    "ingun",
    "ougun",
    "eroop",
    "inoop",
    "ouoop_1",
    "ouoop_2"
]

prev = prev[features].copy()


# =========================
# 3. 2023 → 2024 JOIN
# =========================

df = current[
    [
        "pidwon",
        "annual_medical_cost"
    ]
].merge(
    prev,
    on="pidwon",
    how="inner",
    validate="one_to_one"
)

print("패널 데이터:", df.shape)


# =========================
# 4. X / y 분리
# =========================

X = df.drop(
    columns=[
        "pidwon",
        "annual_medical_cost"
    ]
)

y = df["annual_medical_cost"]


# =========================
# 5. 결측치 처리
# =========================

imputer = SimpleImputer(
    strategy="constant",
    fill_value=0
)

X = imputer.fit_transform(X)


# =========================
# 6. Train / Test 분리
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 7. 최적 Gradient Boosting 모델
# =========================

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=4,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# =========================
# 8. 예측
# =========================

pred = model.predict(X_test)


# =========================
# 9. 성능 평가
# =========================

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

r2 = r2_score(
    y_test,
    pred
)


# =========================
# 10. 결과 출력
# =========================

print("\n=== 최종 Gradient Boosting 모델 ===")

print(f"Train : {len(X_train):,}명")
print(f"Test  : {len(X_test):,}명")

print()

print(f"MAE  : {mae:,.0f}원")
print(f"RMSE : {rmse:,.0f}원")
print(f"R²   : {r2:.4f}")

print("\n=== 모델 비교 ===")

print("v1                : 0.0469")
print("v2                : 0.0516")
print("v3                : 0.0561")
print("v4                : 0.3629")
print("XGBoost           : 0.1594")
print("CatBoost          : 0.2417")
print("Random Forest     : 0.2864")
print(f"최종 GradientBoost: {r2:.4f}")