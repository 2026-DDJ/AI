import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

# =========================
# 데이터
# =========================

raw = pd.read_spss("data/raw/f_ind.sav")
df = pd.read_csv("data/processed/KHP_2024_week5_v3.csv")

raw.columns = raw.columns.str.lower()

# =========================
# 기존 Feature
# =========================

numeric_base = [
    "age", "income", "bmi", "chronic_count",
    "ergun", "ingun", "ougun", "hs6_1"
]

categorical_base = [
    "sex", "region1", "chronic_disease",
    "self_rated_health", "smoking_status",
    "drinking_frequency", "exercise",
    "marr", "edu", "health_ins", "disa_yn", "eco1",
    "hs1", "hs2_yn", "hs3_yn",
    "hs_eq1", "hs_eq2", "hs_eq3",
    "hs_eq4", "hs_eq5", "hs6_yn"
]

# =========================
# 추가 Feature
# =========================

chronic = [
    "cd1_htn","cd1_dm","cd1_dys","cd1_cld","cd1_ald",
    "cd1_lc","cd1_oak","cd1_oae","cd1_ra","cd1_oas",
    "cd1_vd","cd1_vde","cd1_ost","cd1_gc","cd1_cc",
    "cd1_lca","cd1_bc","cd1_cvc","cd1_tc","cd1_etc",
    "cd1_ap","cd1_mi","cd1_ch","cd1_ci","cd1_ast",
    "cd1_pem","cd1_copd","cd1_bpe","cd1_hpot","cd1_hpt",
    "cd1_md","cd1_dmn","cd1_crd","cd1_1"
]

numeric_extra = [
    "ougun_med",
    "ougun_dent",
    "ougun_ort",
    "p2_2",
    "d5",
    "i_phi_n"
]

categorical_extra = chronic + [
    "hs4_yn","hs5_yn","hs7_yn","hs8_yn",
    "p2","p3",
    "s1","s10","s10_1","s11","s11_1","s12","s13",
    "d1_1","d6","d7",
    "se1","ma1",
    "health_ins_np",
    "usc1_yn","usc2","usc3","usc4_yn",
    "i_phi1_yn",
    "edu_stat","hh_rel","live_t_yn"
]

raw_needed = list(set(
    ["pidwon", "ergun", "ingun", "ougun"]
    + numeric_extra
    + categorical_extra
))

extra = raw[raw_needed].copy()

# 이용 횟수 NaN = 이용 없음
count_vars = [
    "ergun","ingun","ougun",
    "ougun_med","ougun_dent","ougun_ort"
]

for col in count_vars:
    if col in extra.columns:
        extra[col] = extra[col].fillna(0)

# 기존 df에 이미 존재하는 열은 중복 merge 방지
merge_cols = ["pidwon"] + [
    c for c in extra.columns
    if c != "pidwon" and c not in df.columns
]

data = df.merge(
    extra[merge_cols],
    on="pidwon",
    how="left",
    validate="one_to_one"
)

# ergun/ingun/ougun이 df에 없으면 raw에서 추가
for col in ["ergun", "ingun", "ougun"]:
    if col not in data.columns:
        temp = raw[["pidwon", col]].copy()
        temp[col] = temp[col].fillna(0)
        data = data.merge(
            temp,
            on="pidwon",
            how="left",
            validate="one_to_one"
        )

numeric_features = [
    c for c in numeric_base + numeric_extra
    if c in data.columns
]

categorical_features = [
    c for c in categorical_base + categorical_extra
    if c in data.columns
]

features = numeric_features + categorical_features

# =========================
# 타깃
# =========================

threshold = data["annual_medical_cost"].median()

y = (
    data["annual_medical_cost"] > threshold
).astype(int)

X = data[features]

print("=== 확장 Feature ===")
print("데이터 수:", len(data))
print("전체 Feature 수:", len(features))
print("Numeric:", len(numeric_features))
print("Categorical:", len(categorical_features))
print(f"의료비 기준: {threshold:,.0f}원")

# =========================
# 전처리
# =========================

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

# 튜닝에서 나온 최적값
model = HistGradientBoostingClassifier(
    learning_rate=0.03,
    max_iter=200,
    max_leaf_nodes=15,
    l2_regularization=0.5,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# =========================
# 5-Fold CV
# =========================

cv = StratifiedKFold(
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
        "accuracy": "accuracy",
        "balanced": "balanced_accuracy",
        "f1": "f1",
        "auc": "roc_auc"
    },
    n_jobs=-1
)

print("\n=== 확장 모델 5-Fold CV ===")

for i in range(5):
    print(
        f"Fold {i+1} | "
        f"Accuracy {scores['test_accuracy'][i]:.4f} | "
        f"Balanced {scores['test_balanced'][i]:.4f} | "
        f"F1 {scores['test_f1'][i]:.4f} | "
        f"AUC {scores['test_auc'][i]:.4f}"
    )

print("\n=== 평균 ===")
print(f"Accuracy    : {scores['test_accuracy'].mean():.4f}")
print(f"Accuracy SD : {scores['test_accuracy'].std():.4f}")
print(f"Balanced    : {scores['test_balanced'].mean():.4f}")
print(f"F1          : {scores['test_f1'].mean():.4f}")
print(f"ROC-AUC     : {scores['test_auc'].mean():.4f}")

print("\n기존 CV Accuracy: 0.8009")
print(
    "변화:",
    f"{(scores['test_accuracy'].mean() - 0.800935) * 100:+.2f}%p"
)
