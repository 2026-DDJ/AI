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

base = pd.read_csv(CSV)
raw = pd.read_spss(RAW)

base.columns = base.columns.str.lower()
raw.columns = raw.columns.str.lower()

# ============================================================
# BUILD FINAL 15 FEATURES
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
    df["indemnity_insurance"]
    .fillna(0)
    .astype(int)
)
df["hospitalized_year"] = (
    df["hospitalized_year"]
    .fillna(0)
    .astype(int)
)

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
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat
        ),
    ])


def reg(seed):
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


def classifier(seed):
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
# 5-FOLD OOF
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

base_pred = np.zeros(len(df))
two_pred = np.zeros(len(df))

fold_base = []
fold_two = []
aucs = []

for fold, (tr, te) in enumerate(cv.split(df), 1):

    Xtr = df.iloc[tr][FEATURES]
    Xte = df.iloc[te][FEATURES]

    ytr = y.iloc[tr]
    yte = y.iloc[te]

    # --------------------------------------------------------
    # SINGLE XGB
    # --------------------------------------------------------

    base_model = Pipeline([
        ("prep", make_prep()),
        ("model", reg(42 + fold))
    ])

    base_model.fit(Xtr, ytr)

    bp = np.maximum(
        base_model.predict(Xte),
        0
    )

    base_pred[te] = bp

    fold_base.append(
        r2_score(yte, bp)
    )

    # --------------------------------------------------------
    # TWO-STAGE P90
    # --------------------------------------------------------

    threshold = np.quantile(ytr, 0.90)

    high_train = (
        ytr >= threshold
    ).astype(int)

    clf = Pipeline([
        ("prep", make_prep()),
        ("model", classifier(100 + fold))
    ])

    clf.fit(
        Xtr,
        high_train
    )

    high_prob = clf.predict_proba(Xte)[:, 1]

    low_mask = ytr < threshold
    high_mask = ytr >= threshold

    low_model = Pipeline([
        ("prep", make_prep()),
        ("model", reg(200 + fold))
    ])

    high_model = Pipeline([
        ("prep", make_prep()),
        ("model", reg(300 + fold))
    ])

    low_model.fit(
        Xtr.loc[low_mask],
        ytr.loc[low_mask]
    )

    high_model.fit(
        Xtr.loc[high_mask],
        ytr.loc[high_mask]
    )

    lp = np.maximum(
        low_model.predict(Xte),
        0
    )

    hp = np.maximum(
        high_model.predict(Xte),
        0
    )

    tp = (
        (1 - high_prob) * lp
        + high_prob * hp
    )

    two_pred[te] = tp

    fold_two.append(
        r2_score(yte, tp)
    )

    auc = roc_auc_score(
        (yte >= threshold).astype(int),
        high_prob
    )

    aucs.append(auc)

# ============================================================
# SINGLE RESULTS
# ============================================================

print("=" * 105)
print("15-FEATURE BASE MODELS")
print("=" * 105)

base_r2 = r2_score(y, base_pred)
two_r2 = r2_score(y, two_pred)

print(
    f"XGB | "
    f"R²={base_r2:.4f} | "
    f"fold={np.mean(fold_base):.4f}±{np.std(fold_base):.4f} | "
    f"MAE={mean_absolute_error(y, base_pred):,.0f} | "
    f"RMSE={np.sqrt(mean_squared_error(y, base_pred)):,.0f}"
)

print(
    f"TWO_P90 | "
    f"R²={two_r2:.4f} | "
    f"fold={np.mean(fold_two):.4f}±{np.std(fold_two):.4f} | "
    f"MAE={mean_absolute_error(y, two_pred):,.0f} | "
    f"RMSE={np.sqrt(mean_squared_error(y, two_pred)):,.0f} | "
    f"AUC={np.mean(aucs):.4f}"
)

# ============================================================
# WEIGHT SEARCH
# ============================================================

print()
print("=" * 105)
print("ENSEMBLE WEIGHT SEARCH")
print("=" * 105)

results = []

for w in np.arange(
    0.0,
    1.0001,
    0.025
):

    pred = (
        w * base_pred
        + (1 - w) * two_pred
    )

    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(
        mean_squared_error(y, pred)
    )

    results.append(
        (
            r2,
            mae,
            rmse,
            w
        )
    )

results.sort(
    reverse=True,
    key=lambda x: x[0]
)

for rank, (
    r2,
    mae,
    rmse,
    w
) in enumerate(
    results[:15],
    1
):

    print(
        f"{rank:2d}. "
        f"{w:.3f} XGB + "
        f"{1-w:.3f} TWO_P90 | "
        f"R²={r2:.4f} | "
        f"MAE={mae:,.0f} | "
        f"RMSE={rmse:,.0f}"
    )

best_r2, best_mae, best_rmse, best_w = (
    results[0]
)

# ============================================================
# HIGH-COST CHECK
# ============================================================

best_pred = (
    best_w * base_pred
    + (1 - best_w) * two_pred
)

p90 = np.quantile(y, 0.90)
high = y >= p90
low = ~high

print()
print("=" * 105)
print("HIGH-COST CHECK")
print("=" * 105)

print(
    f"Overall R² : "
    f"{r2_score(y, best_pred):.4f}"
)

print(
    f"Low 90% MAE: "
    f"{mean_absolute_error(y[low], best_pred[low]):,.0f}"
)

print(
    f"High 10% MAE: "
    f"{mean_absolute_error(y[high], best_pred[high]):,.0f}"
)

print(
    f"High 10% actual mean: "
    f"{y[high].mean():,.0f}"
)

print(
    f"High 10% predicted mean: "
    f"{best_pred[high].mean():,.0f}"
)

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 105)
print("FINAL RESULT")
print("=" * 105)

print("Features:", FEATURES)

print(
    f"Single XGB R² = "
    f"{base_r2:.4f}"
)

print(
    f"Best ensemble = "
    f"{best_w:.3f} XGB + "
    f"{1-best_w:.3f} TWO_P90"
)

print(
    f"Best ensemble R² = "
    f"{best_r2:.4f}"
)

print(
    f"MAE = "
    f"{best_mae:,.0f}원"
)

print(
    f"RMSE = "
    f"{best_rmse:,.0f}원"
)

print(
    f"Previous best = "
    f"0.4508"
)

print(
    f"Gain = "
    f"{best_r2 - 0.4508:+.4f}"
)

print(
    f"Gap to 0.50 = "
    f"{0.50 - best_r2:+.4f}"
)
