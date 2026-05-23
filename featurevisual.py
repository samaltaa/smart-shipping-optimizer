from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess


def load_data():
    path = Path("C:/Users/Grace/mlprojects/data")
    orders = pd.read_csv(path / "olist_orders_dataset.csv")
    order_items = pd.read_csv(path / "olist_order_items_dataset.csv")
    order_payments = pd.read_csv(path / "olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv(path / "olist_order_reviews_dataset.csv")
    products = pd.read_csv(path / "olist_products_dataset.csv")
    sellers = pd.read_csv(path / "olist_sellers_dataset.csv")
    customers = pd.read_csv(path / "olist_customers_dataset.csv")
    geo = pd.read_csv(path / "olist_geolocation_dataset.csv")
    geo = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    df = orders.merge(order_items, on="order_id")
    df = df.merge(products, on="product_id")
    df = df.merge(sellers, on="seller_id")
    df = df.merge(customers, on="customer_id")
    df = df.merge(geo, left_on="seller_zip_code_prefix", right_index=True)
    df = df.rename(columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"})
    df = df.merge(geo, left_on="customer_zip_code_prefix", right_index=True)
    df = df.rename(columns={"geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng"})
    payments_agg = order_payments.groupby("order_id").agg(
        payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", lambda x: x.mode()[0]),
        payment_value=("payment_value", "sum"),
        payment_sequential=("payment_sequential", "max"),
        payment_type_count=("payment_type", "nunique")
    ).reset_index()
    df = df.merge(payments_agg, on="order_id", how="left")
    reviews_agg = order_reviews.groupby("order_id").agg(
        review_score=("review_score", "mean"),
        review_comment_length=("review_comment_message", lambda x: x.dropna().str.len().mean()),
        review_creation_date=("review_creation_date", "min"),
        review_answer_timestamp=("review_answer_timestamp", "min")
    ).reset_index()
    df = df.merge(reviews_agg, on="order_id", how="left")
    return df


logistics = load_data()

logistics["order_purchase_timestamp"] = pd.to_datetime(logistics["order_purchase_timestamp"])
logistics["order_estimated_delivery_date"] = pd.to_datetime(logistics["order_estimated_delivery_date"])
logistics["order_approved_at"] = pd.to_datetime(logistics["order_approved_at"])
logistics["shipping_limit_date"] = pd.to_datetime(logistics["shipping_limit_date"])
logistics["order_delivered_customer_date"] = pd.to_datetime(logistics["order_delivered_customer_date"])
logistics["review_creation_date"] = pd.to_datetime(logistics["review_creation_date"])
logistics["review_answer_timestamp"] = pd.to_datetime(logistics["review_answer_timestamp"])

logistics["estimated_delivery_days"] = (
    logistics["order_estimated_delivery_date"] - logistics["order_purchase_timestamp"]
).dt.days
logistics = logistics.dropna(subset=["estimated_delivery_days"])

logistics["product_volume"] = (
    logistics["product_length_cm"] *
    logistics["product_height_cm"] *
    logistics["product_width_cm"]
)
logistics["purchase_hour"] = logistics["order_purchase_timestamp"].dt.hour
logistics["purchase_dayofweek"] = logistics["order_purchase_timestamp"].dt.dayofweek
logistics["purchase_month"] = logistics["order_purchase_timestamp"].dt.month
logistics["payment_approval_delay"] = (
    logistics["order_approved_at"] - logistics["order_purchase_timestamp"]
).dt.total_seconds() / 3600

category_complexity = {
    "moveis_escritorio": 5, "moveis_decoracao": 5, "moveis_quarto": 5,
    "moveis_sala": 5, "moveis_colchao_e_estofado": 5,
    "eletrodomesticos": 4, "eletrodomesticos_2": 4, "eletroportateis": 4,
    "informatica_acessorios": 3, "eletronicos": 3, "telefonia": 3,
    "esporte_lazer": 3, "brinquedos": 3, "ferramentas_jardim": 3,
    "beleza_saude": 2, "utilidades_domesticas": 2, "cama_mesa_banho": 2,
    "livros_tecnicos": 1, "livros_interesse_geral": 1, "musica": 1,
}
logistics["category_complexity"] = logistics["product_category_name"].map(category_complexity).fillna(2)

order_agg = logistics.groupby("order_id").agg(
    items_per_order=("order_item_id", "count"),
    unique_sellers_per_order=("seller_id", "nunique"),
    total_weight=("product_weight_g", "sum"),
    total_volume=("product_volume", "sum"),
    order_total_price=("price", "sum")
).reset_index()
logistics = logistics.merge(order_agg, on="order_id", how="left")

seller_stats = logistics.groupby("seller_id").agg(
    seller_avg_review=("review_score", "mean"),
    seller_avg_order_value=("price", "mean"),
    seller_total_orders=("order_id", "count"),
    seller_review_volatility=("review_score", "std")
).reset_index()
logistics = logistics.merge(seller_stats, on="seller_id", how="left")

seller_tenure = logistics.groupby("seller_id")["order_purchase_timestamp"].min().reset_index()
seller_tenure.columns = ["seller_id", "seller_first_sale_date"]
logistics = logistics.merge(seller_tenure, on="seller_id", how="left")
logistics["seller_age_days"] = (
    logistics["order_purchase_timestamp"] - logistics["seller_first_sale_date"]
).dt.days

customer_stats = logistics.groupby("customer_unique_id").agg(
    customer_lifetime_value=("price", "sum"),
).reset_index()
logistics = logistics.merge(customer_stats, on="customer_unique_id", how="left")

customer_city_density = logistics.groupby("customer_city")["order_id"].count().reset_index()
customer_city_density.columns = ["customer_city", "customer_city_order_density"]
logistics = logistics.merge(customer_city_density, on="customer_city", how="left")

daily_orders = logistics.groupby(logistics["order_purchase_timestamp"].dt.date).size().reset_index()
daily_orders.columns = ["purchase_date", "daily_order_count"]
daily_orders["purchase_date"] = pd.to_datetime(daily_orders["purchase_date"])
daily_orders = daily_orders.sort_values("purchase_date")
daily_orders["rolling_7d_orders"] = daily_orders["daily_order_count"].rolling(7).mean()
logistics["purchase_date"] = pd.to_datetime(logistics["order_purchase_timestamp"].dt.date)
logistics = logistics.merge(daily_orders[["purchase_date", "daily_order_count", "rolling_7d_orders"]], on="purchase_date", how="left")

logistics["shipping_limit_days"] = (
    logistics["shipping_limit_date"] - logistics["order_purchase_timestamp"]
).dt.days
logistics["seller_processing_window"] = (
    logistics["shipping_limit_date"] - logistics["order_approved_at"]
).dt.total_seconds() / 3600
logistics["shipping_window_x_complexity"] = (
    logistics["shipping_limit_days"] * logistics["category_complexity"]
)
logistics["log_city_density"] = np.log1p(logistics["customer_city_order_density"])
logistics["log_seller_age"] = np.log1p(logistics["seller_age_days"])
logistics["log_review_comment"] = np.log1p(logistics["review_comment_length"].fillna(0))
logistics["log_clv"] = np.log1p(logistics["customer_lifetime_value"])
logistics["log_total_weight"] = np.log1p(logistics["total_weight"])
logistics["log_approval_delay"] = np.log1p(
    logistics["payment_approval_delay"].replace([np.inf, -np.inf], np.nan)
)
logistics["log_installments"] = np.log1p(logistics["payment_installments"])
logistics["log_rolling_7d"] = np.log1p(logistics["rolling_7d_orders"])

logistics["clv_x_complexity"] = logistics["log_clv"] * logistics["category_complexity"]
logistics["freight_burden"] = (
    logistics["freight_value"] / logistics["price"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan) * logistics["log_total_weight"]
logistics["complex_heavy_order"] = logistics["category_complexity"] * logistics["log_total_weight"]
logistics["operational_stress"] = (
    logistics["log_approval_delay"].fillna(0) +
    logistics["unique_sellers_per_order"].fillna(1) +
    logistics["log_installments"].fillna(0)
)
logistics["weight_x_sellers"] = logistics["log_total_weight"] * logistics["unique_sellers_per_order"]
logistics["pressure_x_weight_sellers"] = (
    logistics["rolling_7d_orders"] * logistics["weight_x_sellers"]
).replace([np.inf, -np.inf], np.nan)
logistics["installment_approval_lag"] = (
    logistics["payment_installments"] *
    logistics["log_approval_delay"].replace([np.inf, -np.inf], np.nan)
)
logistics["pressure_x_operational_stress"] = (
    logistics["daily_order_count"] * logistics["operational_stress"]
).replace([np.inf, -np.inf], np.nan)
logistics["high_complexity_installment"] = (
    logistics["category_complexity"] * logistics["payment_installments"]
)
logistics["seller_high_installment_rate"] = logistics["seller_id"].map(
    logistics.groupby("seller_id")["payment_installments"].apply(lambda x: (x > 3).mean())
).fillna(0)
logistics["payment_value_vs_order_value"] = (
    logistics["payment_value"] / logistics["order_total_price"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)


combinations = {
    "shipping_window_x_city_density": (
        logistics["shipping_window_x_complexity"] *
        logistics["log_city_density"]
    ).replace([np.inf, -np.inf], np.nan),

    "processing_window_x_complexity": (
        logistics["seller_processing_window"] *
        logistics["category_complexity"]
    ).replace([np.inf, -np.inf], np.nan),

    "shipping_window_x_seller_age": (
        logistics["shipping_limit_days"] *
        logistics["log_seller_age"]
    ).replace([np.inf, -np.inf], np.nan),

    "city_density_x_clv": (
        logistics["log_city_density"] *
        logistics["log_clv"]
    ).replace([np.inf, -np.inf], np.nan),

    "processing_window_x_weight": (
        logistics["seller_processing_window"] *
        logistics["log_total_weight"]
    ).replace([np.inf, -np.inf], np.nan),

    "shipping_window_x_operational_stress": (
        logistics["shipping_limit_days"] *
        logistics["operational_stress"]
    ).replace([np.inf, -np.inf], np.nan),

    "city_density_x_weight_sellers": (
        logistics["log_city_density"] *
        logistics["weight_x_sellers"]
    ).replace([np.inf, -np.inf], np.nan),

    "seller_age_x_complexity": (
        logistics["log_seller_age"] *
        logistics["category_complexity"]
    ).replace([np.inf, -np.inf], np.nan),

    "processing_window_x_installments": (
        logistics["seller_processing_window"] *
        logistics["log_installments"]
    ).replace([np.inf, -np.inf], np.nan),

    "city_density_x_freight_burden": (
        logistics["log_city_density"] *
        logistics["freight_burden"].replace([np.inf, -np.inf], np.nan)
    ).replace([np.inf, -np.inf], np.nan),

    "shipping_window_x_clv": (
        logistics["shipping_window_x_complexity"] *
        logistics["log_clv"]
    ).replace([np.inf, -np.inf], np.nan),

    "load_x_shipping_window": (
        logistics["log_rolling_7d"] *
        logistics["shipping_limit_days"]
    ).replace([np.inf, -np.inf], np.nan),

    "seller_age_x_review_volatility": (
        logistics["log_seller_age"] *
        logistics["seller_review_volatility"].fillna(0)
    ).replace([np.inf, -np.inf], np.nan),

    "city_density_x_operational_stress": (
        logistics["log_city_density"] *
        logistics["operational_stress"]
    ).replace([np.inf, -np.inf], np.nan),

    "processing_window_x_clv_complexity": (
        logistics["seller_processing_window"] *
        logistics["clv_x_complexity"]
    ).replace([np.inf, -np.inf], np.nan),

    "shipping_window_x_installment_lag": (
        logistics["shipping_limit_days"] *
        logistics["installment_approval_lag"].replace([np.inf, -np.inf], np.nan)
    ).replace([np.inf, -np.inf], np.nan),

    "city_density_x_seller_age": (
        logistics["log_city_density"] *
        logistics["log_seller_age"]
    ).replace([np.inf, -np.inf], np.nan),

    "processing_window_x_seller_age": (
        logistics["seller_processing_window"] *
        logistics["log_seller_age"]
    ).replace([np.inf, -np.inf], np.nan),

    "high_installment_x_shipping_window": (
        logistics["high_complexity_installment"] *
        logistics["shipping_limit_days"]
    ).replace([np.inf, -np.inf], np.nan),

    "payment_complexity_x_processing": (
        logistics["payment_value_vs_order_value"].replace([np.inf, -np.inf], np.nan) *
        logistics["seller_processing_window"]
    ).replace([np.inf, -np.inf], np.nan),
}

print("=" * 60)
print("CROSS-ROUND COMBINATION CORRELATIONS")
print("=" * 60)
print(f"\n{'Feature':<45} {'Pearson r':>10} {'Keep?':>8}")
print("-" * 65)

results = {}
for feat, series in combinations.items():
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    corr = clean.corr(logistics.loc[clean.index, "estimated_delivery_days"])
    results[feat] = corr
    keep = "YES" if abs(corr) > 0.05 else "WEAK"
    print(f"{feat:<45} {corr:>10.4f} {keep:>8}")

print("\n" + "=" * 60)
print("QUINTILE BREAKDOWN")
print("=" * 60)

for feat, corr in sorted(results.items(), key=lambda x: abs(x[1]), reverse=True):
    if abs(corr) < 0.05:
        continue
    logistics[feat] = combinations[feat]
    clean = logistics[[feat, "estimated_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 100:
        continue
    try:
        bins = pd.qcut(clean[feat], 5, duplicates="drop")
        quintile_stats = clean.groupby(bins, observed=True)["estimated_delivery_days"].agg(["mean", "std", "count"])
        spread = quintile_stats["mean"].max() - quintile_stats["mean"].min()
        print(f"\n=== {feat} (r={corr:.4f}, spread={spread:.2f} days) ===")
        print(quintile_stats.to_string())
    except Exception:
        print(f"\n=== {feat} (r={corr:.4f}) === insufficient bins")

fig, axes = plt.subplots(4, 5, figsize=(28, 22))
axes = axes.flatten()

for i, (feat, corr) in enumerate(sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)):
    if i >= 20:
        break
    series = combinations[feat]
    clean = pd.DataFrame({"feat": series, "target": logistics["estimated_delivery_days"]}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 100:
        continue
    sample = clean.sample(min(8000, len(clean)), random_state=42)
    clipped = sample["feat"].clip(lower=sample["feat"].quantile(0.01), upper=sample["feat"].quantile(0.99))
    axes[i].scatter(clipped, sample["target"], alpha=0.05, s=5, color="steelblue")
    try:
        smoothed = lowess(sample["target"], clipped, frac=0.2)
        axes[i].plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=2)
    except Exception:
        pass
    axes[i].set_title(f"{feat[:30]}\nr={corr:.4f}", fontsize=7)
    axes[i].set_xlabel(feat[:25], fontsize=6)
    axes[i].set_ylabel("Days", fontsize=7)

plt.suptitle("Cross-Round Combinations vs Delivery Days", fontsize=13)
plt.tight_layout()
plt.show()

strong = {k: v for k, v in results.items() if abs(v) > 0.05}
for feat, corr in sorted(strong.items(), key=lambda x: abs(x[1]), reverse=True):
    logistics[feat] = combinations[feat]
    clean = logistics[[feat, "estimated_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 100:
        continue
    sample = clean.sample(min(10000, len(clean)), random_state=42)
    clipped = sample[feat].clip(lower=sample[feat].quantile(0.01), upper=sample[feat].quantile(0.99))

    try:
        bins = pd.qcut(clean[feat], 5, duplicates="drop")
        quintile_means = clean.groupby(bins, observed=True)["estimated_delivery_days"].mean()
        spread = quintile_means.max() - quintile_means.min()
    except Exception:
        spread = 0.0
        quintile_means = pd.Series()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].hexbin(clipped, sample["estimated_delivery_days"], gridsize=40, cmap="viridis", bins="log")
    axes[0].set_title(f"Hexbin: {feat[:35]}")
    axes[0].set_xlabel(feat[:30], fontsize=8)
    axes[0].set_ylabel("Delivery Days")
    try:
        smoothed = lowess(sample["estimated_delivery_days"], clipped, frac=0.2)
        axes[1].scatter(clipped, sample["estimated_delivery_days"], alpha=0.05, s=5)
        axes[1].plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=3)
    except Exception:
        axes[1].scatter(clipped, sample["estimated_delivery_days"], alpha=0.05, s=5)
    axes[1].set_title(f"LOWESS: r={corr:.4f}")
    if len(quintile_means) > 0:
        axes[2].bar(range(len(quintile_means)), quintile_means.values, color="steelblue")
        axes[2].set_xticks(range(len(quintile_means)))
        axes[2].set_xticklabels([str(q)[:15] for q in quintile_means.index], rotation=30, ha="right", fontsize=7)
        for j, v in enumerate(quintile_means.values):
            axes[2].text(j, v + 0.1, f"{v:.1f}", ha="center", fontsize=8)
    axes[2].set_title(f"Quintile Means — spread={spread:.2f}d")
    fig.suptitle(f"{feat} | r={corr:.4f} | spread={spread:.2f} days", fontsize=10)
    plt.tight_layout()
    plt.show()

print("\n" + "=" * 60)
print("FINAL RECOMMENDATION")
print("=" * 60)
print(f"\n{'Feature':<45} {'Pearson r':>10} {'Spread':>8} {'Add?':>6}")
print("-" * 71)

for feat, corr in sorted(results.items(), key=lambda x: abs(x[1]), reverse=True):
    logistics[feat] = combinations[feat]
    clean = logistics[[feat, "estimated_delivery_days"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 100:
        spread = 0.0
    else:
        try:
            bins = pd.qcut(clean[feat], 5, duplicates="drop")
            quintile_means = clean.groupby(bins, observed=True)["estimated_delivery_days"].mean()
            spread = quintile_means.max() - quintile_means.min()
        except Exception:
            spread = 0.0
    add = "YES" if abs(corr) > 0.05 and spread > 2.0 else "NO"
    print(f"{feat:<45} {corr:>10.4f} {spread:>8.2f} {add:>6}")