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
# C1 + Q/SCALE SEARCH
# ============================================================

from sklearn.model_selection import KFold

QUANTILES = [0.80, 0.825, 0.85, 0.875, 0.90]
SCALES = [0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25]
SEEDS = [17, 42, 73, 2026, 7777]

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

results = []

print("=" * 110)
print("C1 + QUANTILE / SOFT-GATING SCALE SEARCH")
print("=" * 110)

for Q in QUANTILES:
    for scale in SCALES:

        seed_scores = []
        seed_maes = []
        seed_rmses = []
        seed_aucs = []

        for cv_seed in SEEDS:

            cv = KFold(
                n_splits=5,
                shuffle=True,
                random_state=cv_seed
            )

            pred = np.zeros(len(y))
            fold_aucs = []

            for fold, (tr, te) in enumerate(cv.split(df)):

                Xtr = df.iloc[tr][FEATURES]
                Xte = df.iloc[te][FEATURES]

                ytr = y.iloc[tr]
                yte = y.iloc[te]

                threshold = np.quantile(ytr, Q)

                high_train = (
                    ytr >= threshold
                ).astype(int)

                clf = Pipeline([
                    ("prep", prep()),
                    (
                        "model",
                        clf_model(
                            cv_seed + 1000 + fold
                        )
                    )
                ])

                clf.fit(Xtr, high_train)

                prob = clf.predict_proba(Xte)[:, 1]

                try:
                    fold_aucs.append(
                        roc_auc_score(
                            (yte >= threshold).astype(int),
                            prob
                        )
                    )
                except:
                    pass

                low_mask = ytr < threshold
                high_mask = ytr >= threshold

                low_model = Pipeline([
                    ("prep", prep()),
                    (
                        "model",
                        base_reg(
                            cv_seed + 2000 + fold
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
                            cv_seed + 3000 + fold
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
                    + prob * (high_p * scale)
                )

                pred[te] = np.maximum(
                    fold_pred,
                    0
                )

            seed_scores.append(
                r2_score(y, pred)
            )

            seed_maes.append(
                mean_absolute_error(y, pred)
            )

            seed_rmses.append(
                np.sqrt(
                    mean_squared_error(y, pred)
                )
            )

            seed_aucs.append(
                np.mean(fold_aucs)
            )

        row = {
            "quantile": Q,
            "scale": scale,
            "mean_r2": np.mean(seed_scores),
            "std_r2": np.std(seed_scores),
            "min_r2": np.min(seed_scores),
            "max_r2": np.max(seed_scores),
            "mae": np.mean(seed_maes),
            "rmse": np.mean(seed_rmses),
            "auc": np.mean(seed_aucs)
        }

        results.append(row)

        print(
            f"Q={Q:.3f} | "
            f"SCALE={scale:.2f} | "
            f"R²={row['mean_r2']:.4f} | "
            f"MIN={row['min_r2']:.4f}"
        )

res = pd.DataFrame(results).sort_values(
    ["mean_r2", "min_r2"],
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 110)
print("TOP 15")
print("=" * 110)

print(
    res.head(15).to_string(
        index=False,
        formatters={
            "mean_r2": "{:.4f}".format,
            "std_r2": "{:.4f}".format,
            "min_r2": "{:.4f}".format,
            "max_r2": "{:.4f}".format,
            "mae": "{:,.0f}".format,
            "rmse": "{:,.0f}".format,
            "auc": "{:.4f}".format
        }
    )
)

res.to_csv(
    "data/processed/c1_q_scale_search_results.csv",
    index=False
)

print("\nBEST")
print(res.iloc[0].to_dict())
