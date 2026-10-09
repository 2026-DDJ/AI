import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ============================================================
# 1. DATA
# ============================================================

DATA_PATH = "data/processed/KHP_2024_week5_v3.csv"
TARGET = "annual_medical_cost"

df = pd.read_csv(DATA_PATH)

# ============================================================
# 2. LEAKAGE BLOCK
# ============================================================

FORBIDDEN = {
    TARGET,
    "eroop",
    "inoop",
    "ouoop_1",
    "ouoop_2",
    "ouoop_1_med",
    "ouoop_1_dent",
    "ouoop_1_ort",
    "ouoop_2_med",
    "ouoop_2_dent",
    "ouoop_2_ort",
}

# 사용자가 실제로 입력할 수 있는 후보만
CANDIDATES = [
    "age",
    "sex",
    "region1",
    "income",
    "bmi",
    "chronic_disease",
    "chronic_count",
    "self_rated_health",
    "smoking_status",
    "drinking_frequency",
    "exercise",
    "health_ins",
    "ougun",
    "ingun",
    "ergun",

    # CSV에 있으면 추가 탐색
    "ougun_med",
    "ougun_dent",
    "ougun_ort",
]

features = [
    f for f in CANDIDATES
    if f in df.columns
    and f not in FORBIDDEN
]

print("=" * 80)
print("OVERNIGHT PERFORMANCE SEARCH")
print("=" * 80)
print("Samples:", len(df))
print("Candidate features:", len(features))
print(features)

X = df[features].copy()
y = df[TARGET].copy()

valid = y.notna()
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

# ============================================================
# 3. FEATURE TYPES
# ============================================================

CAT_SET = {
    "sex",
    "region1",
    "chronic_disease",
    "self_rated_health",
    "smoking_status",
    "drinking_frequency",
    "exercise",
    "health_ins",
}

def get_preprocessor(selected):

    num = [
        f for f in selected
        if f not in CAT_SET
    ]

    cat = [
        f for f in selected
        if f in CAT_SET
    ]

    transformers = []

    if num:
        transformers.append(
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                ]),
                num,
            )
        )

    if cat:
        transformers.append(
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]),
                cat,
            )
        )

    return ColumnTransformer(
        transformers
    )


# ============================================================
# 4. CV
# ============================================================

cv3 = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

cv5 = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ============================================================
# 5. FAST FEATURE SEARCH
#
# permutation ranking에서 이미 확인된 순서를 출발점으로 사용
# 핵심 의료이용 변수는 유지하면서 주변 변수 조합 탐색
# ============================================================

ranking = [
    "ingun",
    "ougun",
    "income",
    "health_ins",
    "region1",
    "self_rated_health",
    "bmi",
    "exercise",
    "smoking_status",
    "age",
    "sex",
    "chronic_count",
    "chronic_disease",
    "ergun",
    "drinking_frequency",
]

ranking = [
    f for f in ranking
    if f in features
]

# ranking에 없지만 CSV에 존재하는 후보 추가
for f in features:
    if f not in ranking:
        ranking.append(f)

feature_sets = []

# Top-N
for n in range(
    4,
    len(ranking) + 1
):
    feature_sets.append(
        ranking[:n]
    )

# 핵심 변수 + 나머지 개별 추가/제거
core = [
    f for f in [
        "ingun",
        "ougun",
        "income",
        "health_ins",
    ]
    if f in features
]

others = [
    f for f in features
    if f not in core
]

for f in others:
    feature_sets.append(
        core + [f]
    )

# core + 2개 조합
for i in range(len(others)):
    for j in range(i + 1, len(others)):
        feature_sets.append(
            core + [
                others[i],
                others[j],
            ]
        )

# 중복 제거
unique_sets = []

seen = set()

for fs in feature_sets:

    key = tuple(
        sorted(fs)
    )

    if key not in seen:
        seen.add(key)
        unique_sets.append(fs)

feature_sets = unique_sets

print("\n")
print("=" * 80)
print("STAGE 1 - FEATURE SET SEARCH")
print("=" * 80)

print(
    "Feature combinations:",
    len(feature_sets)
)

feature_results = []

base_model = XGBRegressor(
    n_estimators=450,
    learning_rate=0.03,
    max_depth=2,
    min_child_weight=10,
    subsample=0.85,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=10,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

for i, selected in enumerate(
    feature_sets,
    1,
):

    pipe = Pipeline([
        (
            "prep",
            get_preprocessor(
                selected
            ),
        ),
        (
            "model",
            base_model,
        ),
    ])

    scores = cross_val_score(
        pipe,
        X[selected],
        y,
        cv=cv3,
        scoring="r2",
        n_jobs=1,
    )

    feature_results.append({
        "features": selected,
        "r2": scores.mean(),
        "sd": scores.std(),
    })

    if (
        i % 10 == 0
        or i == len(feature_sets)
    ):
        current_best = max(
            feature_results,
            key=lambda z: z["r2"]
        )

        print(
            f"[{i}/{len(feature_sets)}] "
            f"best R²="
            f"{current_best['r2']:.4f}"
        )


feature_results = sorted(
    feature_results,
    key=lambda z: z["r2"],
    reverse=True,
)

print("\nTOP FEATURE SETS")

for rank, r in enumerate(
    feature_results[:10],
    1,
):
    print(
        f"{rank:2d}. "
        f"R²={r['r2']:.4f} | "
        f"{r['features']}"
    )


# 상위 3개 feature set만 다음 단계로
top_feature_sets = [
    r["features"]
    for r in feature_results[:3]
]


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("STAGE 2 - MODEL COMPARISON")
print("=" * 80)

model_candidates = {

    "XGB": XGBRegressor(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=2,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=10,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    ),

    "RF": RandomForestRegressor(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=10,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=10,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    ),

    "HGB": HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=3,
        random_state=42,
    ),
}

model_results = []

for selected in top_feature_sets:

    for name, model in model_candidates.items():

        pipe = Pipeline([
            (
                "prep",
                get_preprocessor(
                    selected
                ),
            ),
            (
                "model",
                model,
            ),
        ])

        scores = cross_val_score(
            pipe,
            X[selected],
            y,
            cv=cv3,
            scoring="r2",
            n_jobs=1,
        )

        result = {
            "name": name,
            "features": selected,
            "r2": scores.mean(),
            "sd": scores.std(),
        }

        model_results.append(
            result
        )

        print(
            f"{name:10s} | "
            f"R²={result['r2']:.4f} | "
            f"N={len(selected)}"
        )

model_results = sorted(
    model_results,
    key=lambda z: z["r2"],
    reverse=True,
)

best_features = model_results[0][
    "features"
]

print("\nBest feature set entering tuning:")
print(best_features)


# ============================================================
# 7. XGBOOST RANDOM SEARCH
# ============================================================

print("\n")
print("=" * 80)
print("STAGE 3 - XGBOOST DEEP SEARCH")
print("=" * 80)

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline([
    (
        "prep",
        get_preprocessor(
            best_features
        ),
    ),
    (
        "model",
        xgb,
    ),
])

param_dist = {

    "model__n_estimators": [
        300,
        450,
        600,
        800,
        1000,
        1300,
    ],

    "model__learning_rate": [
        0.005,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
    ],

    "model__max_depth": [
        1,
        2,
        3,
        4,
        5,
    ],

    "model__min_child_weight": [
        3,
        5,
        10,
        20,
        40,
        80,
    ],

    "model__subsample": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "model__colsample_bytree": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "model__reg_alpha": [
        0,
        0.01,
        0.1,
        0.5,
        1,
        5,
    ],

    "model__reg_lambda": [
        1,
        3,
        5,
        10,
        20,
        50,
    ],

    "model__gamma": [
        0,
        0.01,
        0.1,
        0.5,
        1,
    ],
}

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,

    # 밤새 돌려도 되니까 넓게
    n_iter=250,

    scoring="r2",
    cv=cv3,

    random_state=42,
    n_jobs=1,
    verbose=1,

    return_train_score=False,
)

search.fit(
    X[best_features],
    y,
)

print("\nBEST RANDOM SEARCH")
print(
    "R²:",
    round(
        search.best_score_,
        4
    )
)

print(
    "Params:",
    search.best_params_
)


# ============================================================
# 8. FINAL 5-FOLD TEST
# ============================================================

print("\n")
print("=" * 80)
print("STAGE 4 - FINAL 5-FOLD")
print("=" * 80)

best_params = {
    k.replace(
        "model__",
        ""
    ): v
    for k, v
    in search.best_params_.items()
}

final_r2 = []
final_mae = []
final_rmse = []

for fold, (
    train_idx,
    test_idx,
) in enumerate(
    cv5.split(X),
    1,
):

    X_train = X.iloc[
        train_idx
    ][best_features]

    X_test = X.iloc[
        test_idx
    ][best_features]

    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]

    model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    final_pipe = Pipeline([
        (
            "prep",
            get_preprocessor(
                best_features
            ),
        ),
        (
            "model",
            model,
        ),
    ])

    final_pipe.fit(
        X_train,
        y_train,
    )

    pred = final_pipe.predict(
        X_test
    )

    pred = np.maximum(
        pred,
        0
    )

    r2 = r2_score(
        y_test,
        pred,
    )

    mae = mean_absolute_error(
        y_test,
        pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred,
        )
    )

    final_r2.append(r2)
    final_mae.append(mae)
    final_rmse.append(rmse)

    print(
        f"Fold {fold} | "
        f"R²={r2:.4f} | "
        f"MAE={mae:,.0f} | "
        f"RMSE={rmse:,.0f}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("OVERNIGHT FINAL RESULT")
print("=" * 80)

print(
    "Features:",
    best_features
)

print(
    f"R²   : "
    f"{np.mean(final_r2):.4f}"
)

print(
    f"R² SD: "
    f"{np.std(final_r2):.4f}"
)

print(
    f"MAE  : "
    f"{np.mean(final_mae):,.0f}원"
)

print(
    f"RMSE : "
    f"{np.mean(final_rmse):,.0f}원"
)

print(
    "\nBest parameters:"
)

for k, v in best_params.items():
    print(
        f"{k}: {v}"
    )