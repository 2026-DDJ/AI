import numpy as np

from preprocessing import y_train, y_test


# Train/Test로 나누기 전과 동일한 전체 y 구성
y = np.concatenate([y_train.to_numpy(), y_test.to_numpy()])


print("=== 연간 의료비(y) 분포 ===")

print(f"데이터 수     : {len(y):,}명")
print(f"0원 인원      : {(y == 0).sum():,}명")
print(f"0원 비율      : {(y == 0).mean() * 100:.2f}%")

print()
print(f"평균          : {np.mean(y):,.0f}원")
print(f"중앙값        : {np.median(y):,.0f}원")
print(f"표준편차      : {np.std(y):,.0f}원")
print(f"최솟값        : {np.min(y):,.0f}원")
print(f"최댓값        : {np.max(y):,.0f}원")

print()
print("=== 의료비 분위수 ===")
print(f"75%           : {np.percentile(y, 75):,.0f}원")
print(f"90%           : {np.percentile(y, 90):,.0f}원")
print(f"95%           : {np.percentile(y, 95):,.0f}원")
print(f"99%           : {np.percentile(y, 99):,.0f}원")

p99 = np.percentile(y, 99)

print()
print(f"상위 1% 기준  : {p99:,.0f}원")
print(f"상위 1% 인원  : {(y > p99).sum():,}명")

from preprocessing import feature_cols, df

print("\n=== 현재 모델 입력 변수 ===")

for i, col in enumerate(feature_cols, 1):
    print(f"{i:2d}. {col}")

print("\n=== 입력 변수 기본 정보 ===")

for col in feature_cols:
    print(f"\n[{col}]")
    print(f"종류: {df[col].dtype}")
    print(f"고유값 수: {df[col].nunique()}")

    if df[col].nunique() <= 20:
        print(df[col].value_counts().sort_index())
        
# =========================
# 의료비와 입력변수 관계 확인
# =========================

print("\n=== 수치형 변수와 의료비 상관관계 ===")

for col in ["age", "income", "bmi"]:
    corr = df[col].corr(df["annual_medical_cost"])
    print(f"{col:10s}: {corr:.4f}")


print("\n=== 범주형 변수별 평균 의료비 ===")

categorical_features = [
    "sex",
    "region1",
    "chronic_disease",
    "self_rated_health",
    "smoking_status",
    "drinking_frequency",
    "exercise"
]

for col in categorical_features:
    print(f"\n[{col}]")
    
    result = (
        df.groupby(col)["annual_medical_cost"]
        .agg(["count", "mean", "median"])
        .round(0)
    )
    
    print(result)