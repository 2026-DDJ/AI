import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# preprocessing.py에서 만들어진 데이터 가져오기
from preprocessing import X_train, X_test, y_train, y_test


# =========================
# 1. Linear Regression 학습
# =========================

model = LinearRegression()

model.fit(X_train, y_train)

print("모델 학습 완료")


# =========================
# 2. 의료비 예측
# =========================

y_pred = model.predict(X_test)


# =========================
# 3. 모델 성능 평가
# =========================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Linear Regression 결과 ===")
print(f"MAE  : {mae:,.0f}원")
print(f"RMSE : {rmse:,.0f}원")
print(f"R²   : {r2:.4f}")


# =========================
# 4. 실제값 / 예측값 확인
# =========================

print("\n=== 예측 결과 예시 ===")

for actual, predicted in zip(y_test.iloc[:5], y_pred[:5]):
    print(
        f"실제 의료비: {actual:,.0f}원 | "
        f"예측 의료비: {predicted:,.0f}원"
    )
    
    # =========================
# 5. Log 변환 Linear Regression
# =========================

# 의료비를 log1p 변환
y_train_log = np.log1p(y_train)

# 모델 학습
log_model = LinearRegression()
log_model.fit(X_train, y_train_log)

# 예측
y_pred_log = log_model.predict(X_test)

# 다시 원 단위 의료비로 복원
y_pred_log_original = np.expm1(y_pred_log)

# 성능 평가
log_mae = mean_absolute_error(y_test, y_pred_log_original)
log_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_log_original)
)
log_r2 = r2_score(y_test, y_pred_log_original)

print("\n=== Log Linear Regression 결과 ===")
print(f"MAE  : {log_mae:,.0f}원")
print(f"RMSE : {log_rmse:,.0f}원")
print(f"R²   : {log_r2:.4f}")

# =========================
# 6. Random Forest Regression
# =========================

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# 모델 학습
rf_model.fit(X_train, y_train)

# 예측
rf_pred = rf_model.predict(X_test)

# 성능 평가
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n=== Random Forest Regression 결과 ===")
print(f"MAE  : {rf_mae:,.0f}원")
print(f"RMSE : {rf_rmse:,.0f}원")
print(f"R²   : {rf_r2:.4f}")

# =========================
# 7. Gradient Boosting Regression
# =========================

from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# 모델 학습
gb_model.fit(X_train, y_train)

# 예측
gb_pred = gb_model.predict(X_test)

# 성능 평가
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print("\n=== Gradient Boosting Regression 결과 ===")
print(f"MAE  : {gb_mae:,.0f}원")
print(f"RMSE : {gb_rmse:,.0f}원")
print(f"R²   : {gb_r2:.4f}")