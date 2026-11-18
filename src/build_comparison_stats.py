import pandas as pd
import json

df = pd.read_csv("data/processed/KHP_2024_week5_v3.csv")

df["income_group"] = pd.cut(
    df["income"],
    bins=[-float("inf"), 2400, 4500, 7500, float("inf")],
    labels=[1, 2, 3, 4],
    right=False
)

def summarize(group):
    return {
        "count": int(group.count()),
        "mean": round(float(group.mean())),
        "median": round(float(group.median()))
    }

stats = {
    "overall": summarize(df["annual_medical_cost"]),
    "income": {},
    "outpatient": {},
    "region_income": {}
}

for income, group in df.groupby("income_group", observed=True):
    stats["income"][str(income)] = summarize(group["annual_medical_cost"])


df["outpatient_group"] = pd.cut(
    df["ougun"],
    bins=[-1, 5, 10, 20, 50, float("inf")],
    labels=[1, 2, 3, 4, 5]
)

for group_id, group in df.groupby("outpatient_group", observed=True):
    stats["outpatient"][str(group_id)] = summarize(
        group["annual_medical_cost"]
    )

for (region, income), group in df.groupby(
    ["region1", "income_group"], observed=True
):
    key = f"{int(region)}_{int(income)}"
    stats["region_income"][key] = summarize(group["annual_medical_cost"])

with open("models/comparison_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("완료: models/comparison_stats.json")
print("전체 평균:", stats["overall"]["mean"])
print("소득 구간 수:", len(stats["income"]))
print("지역·소득 집단 수:", len(stats["region_income"]))
