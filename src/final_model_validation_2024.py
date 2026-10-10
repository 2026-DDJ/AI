
import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score

from xgboost import XGBRegressor, XGBClassifier

RAW = "data/raw/f_ind.sav"

CSV = "data/processed/KHP_2024_week5_v3.csv"

TARGET = "annual_medical_cost"

# 완전히 새로운 seed

SEEDS = [17, 73, 2026]

base = pd.read_csv(CSV)

raw = pd.read_spss(RAW)

base.columns = base.columns.str.lower()

raw.columns = raw.columns.str.lower()

# ============================================================

# FINAL FEATURES

# ============================================================

tmp = raw[

    [

        "pidwon",

        "ougun_dent",

        "i_ffs_yn",

        "in1year",

        "hs_eq3",

        "hs_eq5",

    ]

].copy()

for c in tmp.columns[1:]:

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

ougun = pd.to_numeric(

    df["ougun"],

    errors="coerce"

).fillna(0)

df["dent_share"] = np.where(

    ougun > 0,

    df["ougun_dent"] / ougun.replace(0, np.nan),

    0

)

df["dent_share"] = (

    df["dent_share"]

    .replace([np.inf, -np.inf], 0)

    .fillna(0)

    .clip(0, 1)

)

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

}

y = pd.to_numeric(df[TARGET], errors="coerce")

valid = y.notna()

df = df.loc[valid].reset_index(drop=True)

y = y.loc[valid].reset_index(drop=True)

# ============================================================

# PREPROCESSOR

# ============================================================

def make_prep():

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

def make_reg(seed):

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

def make_classifier(seed):

    return XGBClassifier(

        n_estimators=800,

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

# FIXED FINAL ALGORITHM

#

# 75% XGB

# 25% TWO-STAGE P90

#

# 중요:

# threshold는 각 training fold 안에서만 계산

# ============================================================

def predict_fold(train_idx, test_idx, seed):

    Xtr = df.iloc[train_idx][FEATURES]

    Xte = df.iloc[test_idx][FEATURES]

    ytr = y.iloc[train_idx]

    yte = y.iloc[test_idx]

    # -------------------------

    # Base XGB

    # -------------------------

    base_model = Pipeline([

        ("prep", make_prep()),

        ("model", make_reg(seed))

    ])

    base_model.fit(Xtr, ytr)

    base_pred = np.maximum(

        base_model.predict(Xte),

        0

    )

    # -------------------------

    # P90 classifier

    # threshold from TRAIN only

    # -------------------------

    threshold = np.quantile(ytr, 0.90)

    cls_y = (

        ytr >= threshold

    ).astype(int)

    classifier = Pipeline([

        ("prep", make_prep()),

        ("model", make_classifier(seed))

    ])

    classifier.fit(Xtr, cls_y)

    high_prob = classifier.predict_proba(Xte)[:, 1]

    # -------------------------

    # Two regressors

    # -------------------------

    low_mask = ytr < threshold

    high_mask = ytr >= threshold

    low_model = Pipeline([

        ("prep", make_prep()),

        ("model", make_reg(seed + 100))

    ])

    high_model = Pipeline([

        ("prep", make_prep()),

        ("model", make_reg(seed + 200))

    ])

    low_model.fit(

        Xtr.loc[low_mask],

        ytr.loc[low_mask]

    )

    high_model.fit(

        Xtr.loc[high_mask],

        ytr.loc[high_mask]

    )

    low_pred = np.maximum(

        low_model.predict(Xte),

        0

    )

    high_pred = np.maximum(

        high_model.predict(Xte),

        0

    )

    two_pred = (

        (1 - high_prob) * low_pred

        + high_prob * high_pred

    )

    # FINAL FIXED WEIGHT

    final_pred = (

        0.75 * base_pred

        + 0.25 * two_pred

    )

    auc = roc_auc_score(

        (yte >= threshold).astype(int),

        high_prob

    )

    return final_pred, auc

# ============================================================

# VALIDATION 1

# 3 completely new 5-fold splits

# ============================================================

print("=" * 100)

print("VALIDATION 1 — REPEATED NEW-SEED 5-FOLD CV")

print("=" * 100)

seed_results = []

for seed in SEEDS:

    cv = KFold(

        n_splits=5,

        shuffle=True,

        random_state=seed

    )

    pred = np.zeros(len(df))

    aucs = []

    for fold, (tr, te) in enumerate(cv.split(df), 1):

        p, auc = predict_fold(

            tr,

            te,

            seed + fold

        )

        pred[te] = p

        aucs.append(auc)

    r2 = r2_score(y, pred)

    mae = mean_absolute_error(y, pred)

    rmse = np.sqrt(mean_squared_error(y, pred))

    seed_results.append(r2)

    print(

        f"Seed {seed:4d} | "

        f"R²={r2:.4f} | "

        f"MAE={mae:,.0f} | "

        f"RMSE={rmse:,.0f} | "

        f"P90 AUC={np.mean(aucs):.4f}"

    )

print()

print(

    f"Repeated-CV R² mean = "

    f"{np.mean(seed_results):.4f}"

)

print(

    f"Repeated-CV R² SD   = "

    f"{np.std(seed_results):.4f}"

)

# ============================================================

# VALIDATION 2

# Completely separate 20% holdout

# ============================================================

print()

print("=" * 100)

print("VALIDATION 2 — UNTOUCHED-STYLE 80/20 HOLDOUT")

print("=" * 100)

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(

    indices,

    test_size=0.20,

    random_state=98765

)

hold_pred, hold_auc = predict_fold(

    train_idx,

    test_idx,

    98765

)

y_test = y.iloc[test_idx]

hold_r2 = r2_score(

    y_test,

    hold_pred

)

hold_mae = mean_absolute_error(

    y_test,

    hold_pred

)

hold_rmse = np.sqrt(

    mean_squared_error(

        y_test,

        hold_pred

    )

)

print(f"N train: {len(train_idx):,}")

print(f"N test : {len(test_idx):,}")

print(f"R²     : {hold_r2:.4f}")

print(f"MAE    : {hold_mae:,.0f}원")

print(f"RMSE   : {hold_rmse:,.0f}원")

print(f"P90 AUC: {hold_auc:.4f}")

# ============================================================

# SUMMARY

# ============================================================

print()

print("=" * 100)

print("VALIDATION SUMMARY")

print("=" * 100)

print("Selected-model CV record : R² = 0.4508")

print(

    f"New-seed repeated CV     : "

    f"R² = {np.mean(seed_results):.4f}"

)

print(

    f"Separate 20% holdout     : "

    f"R² = {hold_r2:.4f}"

)

print()

print(

    "If repeated-CV and holdout remain close to 0.4508, "

    "the final model is reasonably stable."

)

