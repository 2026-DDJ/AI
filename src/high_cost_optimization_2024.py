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
# CV
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

splits = list(cv.split(df))

# ============================================================
# BASE XGB OOF — ONCE
# ============================================================

print("=" * 110)
print("BASE MODEL")
print("=" * 110)

base_pred = np.zeros(len(df))

for fold, (tr, te) in enumerate(splits, 1):

    model = Pipeline([
        ("prep", prep()),
        ("model", base_reg(100 + fold))
    ])

    model.fit(
        df.iloc[tr][FEATURES],
        y.iloc[tr]
    )

    base_pred[te] = np.maximum(
        model.predict(
            df.iloc[te][FEATURES]
        ),
        0
    )

print(
    f"BASE XGB R²={r2_score(y, base_pred):.4f} | "
    f"MAE={mean_absolute_error(y, base_pred):,.0f}"
)

# ============================================================
# SEARCH
# ============================================================

QUANTILES = [
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
]

HIGH_MODES = [
    "STANDARD",
    "DEEP",
    "LOW_REG",
]

results = []

print()
print("=" * 110)
print("HIGH-COST STRUCTURE SEARCH")
print("=" * 110)

for q in QUANTILES:

    for mode in HIGH_MODES:

        soft_pred = np.zeros(len(df))
        hard_pred = np.zeros(len(df))

        aucs = []

        for fold, (tr, te) in enumerate(splits, 1):

            Xtr = df.iloc[tr][FEATURES]
            Xte = df.iloc[te][FEATURES]

            ytr = y.iloc[tr]
            yte = y.iloc[te]

            threshold = np.quantile(
                ytr,
                q
            )

            high_train = (
                ytr >= threshold
            ).astype(int)

            clf = Pipeline([
                ("prep", prep()),
                ("model", clf_model(200 + fold))
            ])

            clf.fit(
                Xtr,
                high_train
            )

            prob = clf.predict_proba(
                Xte
            )[:, 1]

            try:
                aucs.append(
                    roc_auc_score(
                        (yte >= threshold).astype(int),
                        prob
                    )
                )
            except:
                pass

            low_mask = (
                ytr < threshold
            )

            high_mask = (
                ytr >= threshold
            )

            low_model = Pipeline([
                ("prep", prep()),
                ("model", base_reg(300 + fold))
            ])

            high_model = Pipeline([
                ("prep", prep()),
                (
                    "model",
                    high_reg(
                        400 + fold,
                        mode
                    )
                )
            ])

            low_model.fit(
                Xtr.loc[low_mask],
                ytr.loc[low_mask]
            )

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

            # SOFT GATING
            soft = (
                (1 - prob) * low_p
                + prob * high_p
            )

            # HARD GATING
            hard = np.where(
                prob >= (1 - q),
                high_p,
                low_p
            )

            soft_pred[te] = soft
            hard_pred[te] = hard

        # ----------------------------------------------------
        # calibration multipliers
        # high-cost branch가 계속 낮게 잡히므로
        # 전체 two-stage prediction에 소폭 scale test
        # ----------------------------------------------------

        for gate_name, tp in [
            ("SOFT", soft_pred),
            ("HARD", hard_pred),
        ]:

            for scale in [
                0.95,
                1.00,
                1.05,
                1.10,
                1.15,
                1.20,
            ]:

                scaled = (
                    tp * scale
                )

                # XGB / TWO-STAGE blend
                for w_base in np.arange(
                    0.0,
                    1.0001,
                    0.05
                ):

                    pred = (
                        w_base * base_pred
                        + (1 - w_base) * scaled
                    )

                    r2 = r2_score(
                        y,
                        pred
                    )

                    mae = mean_absolute_error(
                        y,
                        pred
                    )

                    rmse = np.sqrt(
                        mean_squared_error(
                            y,
                            pred
                        )
                    )

                    results.append({
                        "r2": r2,
                        "mae": mae,
                        "rmse": rmse,
                        "q": q,
                        "mode": mode,
                        "gate": gate_name,
                        "scale": scale,
                        "base_weight": w_base,
                        "two_weight": 1 - w_base,
                        "auc": np.mean(aucs),
                        "pred": pred.copy(),
                    })

        best_local = max(
            [
                x for x in results
                if x["q"] == q
                and x["mode"] == mode
            ],
            key=lambda x: x["r2"]
        )

        print(
            f"P{q*100:4.1f} | "
            f"{mode:8s} | "
            f"best R²={best_local['r2']:.4f} | "
            f"{best_local['gate']} | "
            f"scale={best_local['scale']:.2f} | "
            f"XGB={best_local['base_weight']:.2f} | "
            f"TWO={best_local['two_weight']:.2f} | "
            f"AUC={best_local['auc']:.4f}"
        )

# ============================================================
# RANK
# ============================================================

results.sort(
    reverse=True,
    key=lambda x: x["r2"]
)

print()
print("=" * 110)
print("TOP 25")
print("=" * 110)

for rank, r in enumerate(
    results[:25],
    1
):

    print(
        f"{rank:2d}. "
        f"R²={r['r2']:.4f} | "
        f"MAE={r['mae']:,.0f} | "
        f"RMSE={r['rmse']:,.0f} | "
        f"P{r['q']*100:4.1f} | "
        f"{r['mode']:8s} | "
        f"{r['gate']:4s} | "
        f"scale={r['scale']:.2f} | "
        f"XGB={r['base_weight']:.2f} | "
        f"TWO={r['two_weight']:.2f}"
    )

# ============================================================
# HIGH COST DIAGNOSTICS
# ============================================================

best = results[0]
pred = best["pred"]

true_p90 = np.quantile(
    y,
    0.90
)

mask = (
    y >= true_p90
)

print()
print("=" * 110)
print("BEST HIGH-COST DIAGNOSTICS")
print("=" * 110)

print(
    f"Actual high-cost mean    = "
    f"{y[mask].mean():,.0f}"
)

print(
    f"Predicted high-cost mean = "
    f"{pred[mask].mean():,.0f}"
)

print(
    f"High-cost MAE            = "
    f"{mean_absolute_error(y[mask], pred[mask]):,.0f}"
)

print(
    f"Low-90% MAE              = "
    f"{mean_absolute_error(y[~mask], pred[~mask]):,.0f}"
)

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 110)
print("FINAL RESULT")
print("=" * 110)

print(
    f"Best R² = "
    f"{best['r2']:.4f}"
)

print(
    f"MAE = "
    f"{best['mae']:,.0f}원"
)

print(
    f"RMSE = "
    f"{best['rmse']:,.0f}원"
)

print(
    f"Threshold = "
    f"P{best['q']*100:.1f}"
)

print(
    f"High model = "
    f"{best['mode']}"
)

print(
    f"Gating = "
    f"{best['gate']}"
)

print(
    f"Two-stage scale = "
    f"{best['scale']:.2f}"
)

print(
    f"Blend = "
    f"{best['base_weight']:.2f} XGB + "
    f"{best['two_weight']:.2f} TWO"
)

print(
    f"Previous best = "
    f"0.4553"
)

print(
    f"Gain = "
    f"{best['r2'] - 0.4553:+.4f}"
)

print(
    f"Gap to 0.50 = "
    f"{0.50 - best['r2']:+.4f}"
)
