import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ind = pd.read_spss("data/raw/f_ind.sav")
df = pd.read_csv("data/processed/KHP_2024_week5_v3.csv")

# 의료이용 횟수 추가
util = ind[
    ["pidwon", "ergun", "ingun", "ougun"]
].copy()

util[["ergun", "ingun", "ougun"]] = (
    util[["ergun", "ingun", "ougun"]].fillna(0)
)

data = df.merge(
    util,
    on="pidwon",
    how="left",
    validate="one_to_one"
)

# 수치형 변수
numeric_features = [
    "age",
    "income",
    "bmi",
    "chronic_count",
    "ergun",
    "ingun",
    "ougun",
    "hs6_1"
]

# 범주형 변수
categorical_features = [
    "sex",
    "region1",
    "chronic_disease",
    "self_rated_health",
    "smoking_status",
    "drinking_frequency",
    "exercise",
    "marr",
    "edu",
    "health_ins",
    "disa_yn",
    "eco1",
    "hs1",
    "hs2_yn",
    "hs3_yn",
    "hs_eq1",
    "hs_eq2",
    "hs_eq3",
    "hs_eq4",
    "hs_eq5",
    "hs6_yn"
]

features = numeric_features + categorical_features

X = data[features]
y = data["annual_medical_cost"]

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("=== 2024 확장 Feature 모델 ===")
print(f"데이터 수 : {len(data):,}명")
print(f"Feature 수: {len(features)}")
print(f"MAE       : {mae:,.0f}원")
print(f"RMSE      : {rmse:,.0f}원")
print(f"R²        : {r2:.4f}")

# =========================
# 5-Fold Cross Validation
# =========================

from sklearn.model_selection import KFold, cross_validate

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring={
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error"
    },
    n_jobs=-1
)

cv_r2 = scores["test_r2"]
cv_mae = -scores["test_mae"]
cv_rmse = np.sqrt(-scores["test_mse"])

print("\n=== 5-Fold Cross Validation ===")

for i in range(5):
    print(
        f"Fold {i+1} | "
        f"R² {cv_r2[i]:.4f} | "
        f"MAE {cv_mae[i]:,.0f}원 | "
        f"RMSE {cv_rmse[i]:,.0f}원"
    )

print("\n=== 평균 ===")
print(f"R²   : {cv_r2.mean():.4f}")
print(f"R² SD: {cv_r2.std():.4f}")
print(f"MAE  : {cv_mae.mean():,.0f}원")
print(f"RMSE : {cv_rmse.mean():,.0f}원")

# =========================
# Gradient Boosting 파라미터 비교
# =========================

from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.03, 0.05],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_leaf": [5, 10, 20]
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X, y)

print("\n=== Gradient Boosting 튜닝 결과 ===")
print("Best R²:", round(grid.best_score_, 4))
print("Best Parameters:")

for key, value in grid.best_params_.items():
    print(f"{key}: {value}")

# =========================
# 튜닝 모델 Holdout 평가
# =========================

best_model = grid.best_estimator_

best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)

best_mae = mean_absolute_error(y_test, best_pred)
best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))
best_r2 = r2_score(y_test, best_pred)

print("\n=== 튜닝 모델 Holdout 결과 ===")
print(f"MAE  : {best_mae:,.0f}원")
print(f"RMSE : {best_rmse:,.0f}원")
print(f"R²   : {best_r2:.4f}")
