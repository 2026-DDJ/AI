import pandas as pd

# =========================
# 1. 기존 학습 데이터
# =========================

base = pd.read_csv(
    "data/processed/KHP_2024_week3_final_corrected.csv"
)

# =========================
# 2. KHP 원본 IND 데이터
# =========================

ind = pd.read_spss(
    "data/raw/f_ind.sav"
)

print("=== 데이터 크기 ===")
print("기존 데이터:", base.shape)
print("IND 원본:", ind.shape)

# =========================
# 3. pidwon 확인
# =========================

print("\n=== pidwon 정보 ===")
print("기존 데이터 고유 ID:", base["pidwon"].nunique())
print("IND 고유 ID:", ind["pidwon"].nunique())

print("기존 데이터 중복 ID:", base["pidwon"].duplicated().sum())
print("IND 중복 ID:", ind["pidwon"].duplicated().sum())

# =========================
# 4. ID 매칭 확인
# =========================

matched = base["pidwon"].isin(ind["pidwon"])

print("\n=== ID 매칭 결과 ===")
print(f"매칭 성공: {matched.sum():,}명")
print(f"매칭 실패: {(~matched).sum():,}명")
print(f"매칭률: {matched.mean() * 100:.2f}%")

# =========================
# 5. 추가 Feature 추출
# =========================

# cd1_ 로 시작하는 개별 만성질환 변수
disease_cols = [
    col for col in ind.columns
    if col.startswith("cd1_")
]

print("\n=== 만성질환 변수 ===")
print("개수:", len(disease_cols))
print(disease_cols)


# =========================
# 6. 필요한 IND 변수 선택
# =========================

extra_cols = [
    "pidwon",
    "marr",
    "edu",
    "health_ins",
    "disa_yn",
    "eco1"
] + disease_cols

extra = ind[extra_cols].copy()


# =========================
# 7. 만성질환 개수 생성
# =========================

# 각 질환 변수가 어떤 값으로 코딩되어 있는지 먼저 확인
print("\n=== CD1 변수 값 확인 ===")

for col in disease_cols:
    print(col, extra[col].value_counts(dropna=False).to_dict())
    
# =========================
# 8. 질환 변수 정리
# =========================

# cd1_1은 별도 항목이므로 우선 제외
disease_cols_clean = [
    col for col in disease_cols
    if col != "cd1_1"
]

# 1 = 질환 있음 → 1
# 2 = 질환 없음 → 0
for col in disease_cols_clean:
    extra[col] = extra[col].map({
        1.0: 1,
        2.0: 0
    })

# 개인별 만성질환 개수
extra["chronic_count"] = extra[disease_cols_clean].sum(
    axis=1,
    min_count=1
)

print("\n=== 만성질환 개수 ===")
print(extra["chronic_count"].describe())
print(extra["chronic_count"].value_counts().sort_index())


# =========================
# 9. 기존 데이터와 JOIN
# =========================

add_cols = [
    "pidwon",
    "marr",
    "edu",
    "health_ins",
    "disa_yn",
    "eco1",
    "chronic_count"
] + disease_cols_clean

v2 = base.merge(
    extra[add_cols],
    on="pidwon",
    how="left",
    validate="one_to_one"
)

print("\n=== v2 데이터 ===")
print("크기:", v2.shape)
print("기존 의료비 결측:", v2["annual_medical_cost"].isna().sum())
print("chronic_count 결측:", v2["chronic_count"].isna().sum())


# =========================
# 10. 저장
# =========================

OUTPUT_PATH = "data/processed/KHP_2024_week5_v2.csv"

v2.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n저장 완료:", OUTPUT_PATH)