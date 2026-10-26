import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

RAW = "data/raw/f_ind.sav"
CSV = "data/processed/KHP_2024_week5_v3.csv"
TARGET = "annual_medical_cost"
SEED = 42

MS_CSV = "data/processed/KHP_2024_ms_features.csv"

# ============================================================
# LOAD + FINAL 15 FEATURES
# ============================================================

base = pd.read_csv(CSV)
raw = pd.read_spss(RAW)

base.columns = base.columns.str.lower()
raw.columns = raw.columns.str.lower()

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
df["indemnity_insurance"] = df["indemnity_insurance"].fillna(0).astype(int)
df["hospitalized_year"] = df["hospitalized_year"].fillna(0).astype(int)

ougun = pd.to_numeric(df["ougun"], errors="coerce").fillna(0)
dent = pd.to_numeric(df["ougun_dent"], errors="coerce").fillna(0)

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
# MS EVENT-LEVEL FEATURES
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
    
    # MS event-level features
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

y = pd.to_numeric(df[TARGET], errors="coerce")
valid = y.notna()

df = df.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

# ============================================================
# HELPERS
# ============================================================

def prep():

    cat = [c for c in FEATURES if c in CATEGORICAL]
    num = [c for c in FEATURES if c not in CATEGORICAL]

    return ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median"))
            ]),
            num
        ),
        (
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat
        ),
    ])


def base_reg(seed):

    return XGBRegressor(
        n_estimators=1600,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=2,
        subsample=0.55,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=10,
        gamma=0.5,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1
    )


def high_reg(seed, mode):

    if mode == "STANDARD":
        return XGBRegressor(
            n_estimators=1800,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=2,
            subsample=0.65,
            colsample_bytree=0.7,
            reg_alpha=1,
            reg_lambda=10,
            gamma=0.5,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1
        )

    if mode == "DEEP":
        return XGBRegressor(
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
            random_state=seed,
            n_jobs=-1
        )

    if mode == "LOW_REG":
        return XGBRegressor(
            n_estimators=1800,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=1,
            subsample=0.7,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=3,
            gamma=0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1
        )


def clf_model(seed):

    return XGBClassifier(
        n_estimators=900,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=5,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1
    )

# ============================================================

# ============================================================
# LOW-COST MODEL PARAMETER SEARCH
# C1 HIGH FIXED / Q=.85 / SCALE=1.05 / RAW TARGET
# ============================================================

from sklearn.model_selection import KFold
from itertools import product

PARAMS = []

for depth, child, subsample, colsample, alpha, lam, gamma in product(
    [2, 3, 4, 5],
    [1, 2, 3, 5],
    [0.55, 0.70, 0.85],
    [0.55, 0.70, 0.85],
    [0.0, 0.5, 1.0],
    [3, 5, 10, 15],
    [0.0, 0.5]
):
    PARAMS.append({
        "max_depth": depth,
        "min_child_weight": child,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "reg_alpha": alpha,
        "reg_lambda": lam,
        "gamma": gamma
    })

rng = np.random.RandomState(2027)

idx = rng.choice(
    len(PARAMS),
    size=min(180, len(PARAMS)),
    replace=False
)

PARAMS = [PARAMS[i] for i in idx]

# 현재 base_reg 설정도 반드시 포함
PARAMS.insert(0, {
    "max_depth": 3,
    "min_child_weight": 2,
    "subsample": 0.55,
    "colsample_bytree": 0.60,
    "reg_alpha": 1.0,
    "reg_lambda": 10,
    "gamma": 0.5
})

def candidate_low(seed, p):
    return XGBRegressor(
        n_estimators=1600,
        learning_rate=0.01,
        max_depth=p["max_depth"],
        min_child_weight=p["min_child_weight"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_alpha=p["reg_alpha"],
        reg_lambda=p["reg_lambda"],
        gamma=p["gamma"],
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1
    )

def c1_high(seed):
    return XGBRegressor(
        n_estimators=1800,
        learning_rate=0.008,
        max_depth=5,
        min_child_weight=5,
        subsample=0.60,
        colsample_bytree=0.60,
        reg_alpha=0.5,
        reg_lambda=5,
        gamma=0.5,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1
    )

Q = 0.85
SCALE = 1.05
SEARCH_SEED = 42

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=SEARCH_SEED
)

splits = list(cv.split(df))
results = []

print("=" * 110)
print(f"LOW-COST MODEL PARAMETER SEARCH | {len(PARAMS)} CANDIDATES")
print("C1 HIGH FIXED | Q=.85 | SCALE=1.05")
print("=" * 110)

for num, p in enumerate(PARAMS, 1):

    pred = np.zeros(len(y))

    for fold, (tr, te) in enumerate(splits):

        Xtr = df.iloc[tr][FEATURES]
        Xte = df.iloc[te][FEATURES]

        ytr = y.iloc[tr]

        threshold = np.quantile(ytr, Q)

        high_train = (
            ytr >= threshold
        ).astype(int)

        clf = Pipeline([
            ("prep", prep()),
            (
                "model",
                clf_model(
                    SEARCH_SEED + 1000 + fold
                )
            )
        ])

        clf.fit(Xtr, high_train)

        prob = clf.predict_proba(Xte)[:, 1]

        low_mask = ytr < threshold
        high_mask = ytr >= threshold

        low_model = Pipeline([
            ("prep", prep()),
            (
                "model",
                candidate_low(
                    SEARCH_SEED + 2000 + fold,
                    p
                )
            )
        ])

        low_model.fit(
            Xtr.loc[low_mask],
            ytr.loc[low_mask]
        )

        high_model = Pipeline([
            ("prep", prep()),
            (
                "model",
                c1_high(
                    SEARCH_SEED + 3000 + fold
                )
            )
        ])

        high_model.fit(
            Xtr.loc[high_mask],
            ytr.loc[high_mask]
        )

        low_p = np.maximum(
            low_model.predict(Xte),
            0
        )

        high_p = np.maximum(
            high_model.predict(Xte),
            0
        )

        fold_pred = (
            (1 - prob) * low_p
            + prob * (high_p * SCALE)
        )

        pred[te] = np.maximum(
            fold_pred,
            0
        )

    score = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))

    row = dict(p)
    row.update({
        "r2": score,
        "mae": mae,
        "rmse": rmse
    })

    results.append(row)

    if num == 1 or num % 10 == 0:
        print(
            f"{num:3d}/{len(PARAMS)} | "
            f"R²={score:.4f} | "
            f"BEST={max(x['r2'] for x in results):.4f}"
        )

res = pd.DataFrame(results).sort_values(
    "r2",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 110)
print("TOP 20")
print("=" * 110)

print(
    res.head(20).to_string(
        index=False,
        formatters={
            "r2": "{:.4f}".format,
            "mae": "{:,.0f}".format,
            "rmse": "{:,.0f}".format
        }
    )
)

res.to_csv(
    "data/processed/low_model_search_results.csv",
    index=False
)

print("\nBEST PARAMS")
print(res.iloc[0].to_dict())
