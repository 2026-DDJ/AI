import os
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier


# ============================================================
# PATH
# ============================================================

RAW = "data/raw/f_ind.sav"
CSV = "data/processed/KHP_2024_week5_v3.csv"
MS_CSV = "data/processed/KHP_2024_ms_features.csv"

TARGET = "annual_medical_cost"
SEED = 42

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

base = pd.read_csv(CSV)
raw = pd.read_spss(RAW)

base.columns = base.columns.str.lower()
raw.columns = raw.columns.str.lower()


# ============================================================
# FINAL 15 FEATURES
# ============================================================

need = [
    "pidwon",
    "ougun_dent",
    "i_ffs_yn",
    "in1year",
    "hs_eq3",
    "hs_eq5",
    "eco9",
]

tmp = raw[need].copy()

for c in need[1:]:
    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

tmp["ougun_dent"] = tmp["ougun_dent"].fillna(0)

tmp["indemnity_insurance"] = (
    tmp["i_ffs_yn"] == 1
).astype(int)

tmp["hospitalized_year"] = np.where(
    tmp["in1year"].isin([0, 1]),
    tmp["in1year"],
    0
).astype(int)

tmp["activity_limitation"] = (
    tmp["hs_eq3"]
    .where(tmp["hs_eq3"].isin([1, 2, 3]))
)

tmp["anxiety_depression"] = (
    tmp["hs_eq5"]
    .where(tmp["hs_eq5"].isin([1, 2, 3]))
)

keep = [
    "pidwon",
    "ougun_dent",
    "indemnity_insurance",
    "hospitalized_year",
    "activity_limitation",
    "anxiety_depression",
    "eco9",
]

for c in keep[1:]:
    if c in base.columns:
        base = base.drop(columns=c)

df = base.merge(
    tmp[keep],
    on="pidwon",
    how="left",
    validate="one_to_one"
)

df["ougun_dent"] = df["ougun_dent"].fillna(0)
df["indemnity_insurance"] = (
    df["indemnity_insurance"].fillna(0).astype(int)
)
df["hospitalized_year"] = (
    df["hospitalized_year"].fillna(0).astype(int)
)


# ============================================================
# DENT SHARE
# ============================================================

ougun = pd.to_numeric(
    df["ougun"],
    errors="coerce"
).fillna(0)

dent = pd.to_numeric(
    df["ougun_dent"],
    errors="coerce"
).fillna(0)

df["dent_share"] = np.where(
    ougun > 0,
    dent / ougun.replace(0, np.nan),
    0
)

df["dent_share"] = (
    df["dent_share"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
    .clip(0, 1)
)


# ============================================================
# MS FEATURES
# ============================================================

ms = pd.read_csv(MS_CSV)
ms.columns = ms.columns.str.lower()

MS_FEATURES = [
    "pidwon",
    "ms_total_days",
    "ms_in_treatment_nunique",
    "ms_facility_nunique",
    "ms_mtype_1_count",
    "ms_mtype_3_count",
    "ms_in_reason_2_count",
    "ms_in_trt_1_count",
    "ms_in_choice_1_count",
    "ms_in_care2_2_count",
    "ms_in_dsc_nunique",
    "ms_dent_trt_10_count",
    "ms_dent_trt_5_count",
    "ms_in_sat7_mean",
]

ms = ms[MS_FEATURES].copy()

for c in MS_FEATURES[1:]:
    ms[c] = pd.to_numeric(
        ms[c],
        errors="coerce"
    ).fillna(0)

df = df.merge(
    ms,
    on="pidwon",
    how="left",
    validate="one_to_one"
)

for c in MS_FEATURES[1:]:
    df[c] = df[c].fillna(0)


# ============================================================
# FINAL 28 FEATURES
# ============================================================

FEATURES = [
    "ingun",
    "income",
    "health_ins",
    "region1",
    "self_rated_health",
    "bmi",
    "exercise",
    "ougun",
    "ougun_dent",
    "dent_share",
    "indemnity_insurance",
    "hospitalized_year",
    "activity_limitation",
    "anxiety_depression",
    "eco9",

    "ms_total_days",
    "ms_in_treatment_nunique",
    "ms_facility_nunique",
    "ms_mtype_1_count",
    "ms_mtype_3_count",
    "ms_in_reason_2_count",
    "ms_in_trt_1_count",
    "ms_in_choice_1_count",
    "ms_in_care2_2_count",
    "ms_in_dsc_nunique",
    "ms_dent_trt_10_count",
    "ms_dent_trt_5_count",
    "ms_in_sat7_mean",
]

CATEGORICAL = {
    "health_ins",
    "region1",
    "self_rated_health",
    "exercise",
    "indemnity_insurance",
    "hospitalized_year",
    "activity_limitation",
    "anxiety_depression",
    "eco9",
}


# ============================================================
# DATA
# ============================================================

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

valid = y.notna()

df = df.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

X = df[FEATURES]


# ============================================================
# PREPROCESSING
# ============================================================

def prep():

    cat = [
        c for c in FEATURES
        if c in CATEGORICAL
    ]

    num = [
        c for c in FEATURES
        if c not in CATEGORICAL
    ]

    return ColumnTransformer([
        (
            "num",
            Pipeline([
                (
                    "imp",
                    SimpleImputer(strategy="median")
                )
            ]),
            num
        ),
        (
            "cat",
            Pipeline([
                (
                    "imp",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "ohe",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                ),
            ]),
            cat
        ),
    ])


# ============================================================
# HIGH-COST THRESHOLD
# ============================================================

Q = 0.85
SCALE = 1.05

threshold = np.quantile(y, Q)

high_target = (
    y >= threshold
).astype(int)

low_mask = y < threshold
high_mask = y >= threshold

print("=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)
print(f"Samples       : {len(X):,}")
print(f"Features      : {len(FEATURES)}")
print(f"P85 threshold : {threshold:,.0f}원")
print(f"Low samples   : {low_mask.sum():,}")
print(f"High samples  : {high_mask.sum():,}")


# ============================================================
# 1. HIGH-COST CLASSIFIER
# ============================================================

classifier = Pipeline([
    ("prep", prep()),
    (
        "model",
        XGBClassifier(
            n_estimators=900,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=3,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=5,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1
        )
    )
])

print("\n[1/3] Training classifier...")

classifier.fit(
    X,
    high_target
)


# ============================================================
# 2. LOW-COST REGRESSOR
# ============================================================

low_model = Pipeline([
    ("prep", prep()),
    (
        "model",
        XGBRegressor(
            n_estimators=1600,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=2,
            subsample=0.55,
            colsample_bytree=0.60,
            reg_alpha=1.0,
            reg_lambda=10,
            gamma=0.5,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=-1
        )
    )
])

print("[2/3] Training low-cost model...")

low_model.fit(
    X.loc[low_mask],
    y.loc[low_mask]
)


# ============================================================
# 3. HIGH-COST REGRESSOR
# ============================================================

high_model = Pipeline([
    ("prep", prep()),
    (
        "model",
        XGBRegressor(
            n_estimators=1800,
            learning_rate=0.008,
            max_depth=4,
            min_child_weight=3,
            subsample=0.7,
            colsample_bytree=0.75,
            reg_alpha=1,
            reg_lambda=15,
            gamma=0.5,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=-1
        )
    )
])

print("[3/3] Training high-cost model...")

high_model.fit(
    X.loc[high_mask],
    y.loc[high_mask]
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    classifier,
    f"{MODEL_DIR}/classifier.pkl"
)

joblib.dump(
    low_model,
    f"{MODEL_DIR}/low_model.pkl"
)

joblib.dump(
    high_model,
    f"{MODEL_DIR}/high_model.pkl"
)

metadata = {
    "features": FEATURES,
    "threshold": float(threshold),
    "q": Q,
    "scale": SCALE,
}

joblib.dump(
    metadata,
    f"{MODEL_DIR}/metadata.pkl"
)

print("\n" + "=" * 70)
print("FINAL MODEL SAVED")
print("=" * 70)

print("models/classifier.pkl")
print("models/low_model.pkl")
print("models/high_model.pkl")
print("models/metadata.pkl")

print("\nDONE")