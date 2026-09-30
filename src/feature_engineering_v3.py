import pandas as pd


# =========================
# 1. 데이터 불러오기
# =========================

v2 = pd.read_csv(
    "data/processed/KHP_2024_week5_v2.csv"
)

ind = pd.read_spss(
    "data/raw/f_ind.sav"
)


# =========================
# 2. v3 Feature
# =========================

v3_features = [
    "hs1",
    "hs2_yn",
    "hs3_yn",
    "hs_eq1",
    "hs_eq2",
    "hs_eq3",
    "hs_eq4",
    "hs_eq5",
    "hs6_yn",
    "hs6_1"
]

health = ind[
    ["pidwon"] + v3_features
].copy()


# =========================
# 3. 와병 일수 처리
# =========================

# 와병하지 않았다고 응답한 경우
# 와병 일수 = 0일
health.loc[
    health["hs6_yn"] == 2,
    "hs6_1"
] = 0


# =========================
# 4. v2 + 건강상태 Feature JOIN
# =========================

v3 = v2.merge(
    health,
    on="pidwon",
    how="left",
    validate="one_to_one"
)


# =========================
# 5. 결과 확인
# =========================

print("=== v3 데이터 ===")
print("크기:", v3.shape)

print("\n=== 추가 Feature 결측률 ===")

for col in v3_features:
    print(
        f"{col:10s}: "
        f"{v3[col].isna().mean() * 100:.2f}%"
    )


# =========================
# 6. 저장
# =========================

OUTPUT_PATH = (
    "data/processed/"
    "KHP_2024_week5_v3.csv"
)

v3.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n저장 완료:", OUTPUT_PATH)