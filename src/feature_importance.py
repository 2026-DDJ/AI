import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score


# =========================
# 1. 데이터 불러오기
# =========================

prev = pd.read_spss("data/raw/e_ind.sav")

current = pd.read_csv(
    "data/processed/KHP_2024_week5_v3.csv"
)


# =========================
# 2. Feature 선택
# =========================

features = [
    "ergun",
    "ingun",
    "ougun",
    "eroop",
    "inoop",
    "ouoop_1",
    "ouoop_2"
]

prev = prev[
    ["pidwon"] + features
].copy()


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
# 4. X / y
# =========================

X = df[features]

y = df["annual_medical_cost"]


# =========================
# 5. 결측치 처리
# =========================

imputer = SimpleImputer(
    strategy="constant",
    fill_value=0
)

X_imputed = imputer.fit_transform(X)


# =========================
# 6. Train / Test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 7. 최종 모델 학습
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
# 8. 성능 확인
# =========================

pred = model.predict(X_test)

r2 = r2_score(
    y_test,
    pred
)

print("\n=== 모델 성능 ===")
print(f"R² : {r2:.4f}")


# =========================
# 9. Feature Importance
# =========================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)

importance["importance_percent"] = (
    importance["importance"] * 100
)


# =========================
# 10. 결과 출력
# =========================

print("\n=== Feature Importance ===")

for i, row in importance.iterrows():

    print(
        f"{i + 1}. "
        f"{row['feature']:<10} "
        f": {row['importance_percent']:.2f}%"
    )


# =========================
# 11. CSV 저장
# =========================

importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

print(
    "\n저장 완료: "
    "data/processed/feature_importance.csv"
)