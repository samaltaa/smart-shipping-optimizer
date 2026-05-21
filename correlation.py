from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from statsmodels.nonparametric.smoothers_lowess import lowess


def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    lat1, lon1, lat2, lon2 = map(
        radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_data():
    path = Path("C:/Users/Grace/mlprojects/data")

    orders = pd.read_csv(path / "olist_orders_dataset.csv")
    order_items = pd.read_csv(path / "olist_order_items_dataset.csv")
    products = pd.read_csv(path / "olist_products_dataset.csv")
    sellers = pd.read_csv(path / "olist_sellers_dataset.csv")
    customers = pd.read_csv(path / "olist_customers_dataset.csv")
    geo = pd.read_csv(path / "olist_geolocation_dataset.csv")

    geo = geo.groupby(
        "geolocation_zip_code_prefix"
    )[["geolocation_lat", "geolocation_lng"]].mean()

    df = orders.merge(order_items, on="order_id")
    df = df.merge(products, on="product_id")
    df = df.merge(sellers, on="seller_id")
    df = df.merge(customers, on="customer_id")

    df = df.merge(
        geo,
        left_on="seller_zip_code_prefix",
        right_index=True
    )

    df = df.rename(
        columns={
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng"
        }
    )

    df = df.merge(
        geo,
        left_on="customer_zip_code_prefix",
        right_index=True
    )

    df = df.rename(
        columns={
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng"
        }
    )

    return df


logistics = load_data()


logistics["order_purchase_timestamp"] = pd.to_datetime(
    logistics["order_purchase_timestamp"]
)

logistics["order_estimated_delivery_date"] = pd.to_datetime(
    logistics["order_estimated_delivery_date"]
)

logistics["estimated_delivery_days"] = (
    logistics["order_estimated_delivery_date"]
    - logistics["order_purchase_timestamp"]
).dt.days


logistics["real_distance_km"] = logistics.apply(
    lambda row: haversine(
        row["seller_lat"],
        row["seller_lng"],
        row["customer_lat"],
        row["customer_lng"]
    ),
    axis=1
)

logistics["same_state"] = (
    logistics["seller_state"]
    == logistics["customer_state"]
).astype(int)

logistics["state_pair"] = (
    logistics["seller_state"]
    + "_"
    + logistics["customer_state"]
)

logistics["product_volume"] = (
    logistics["product_length_cm"]
    * logistics["product_height_cm"]
    * logistics["product_width_cm"]
)

logistics["distance_squared"] = (
    logistics["real_distance_km"] ** 2
)

logistics["log_distance"] = np.log1p(
    logistics["real_distance_km"]
)

logistics["distance_weight"] = (
    logistics["real_distance_km"]
    * logistics["product_weight_g"]
)

logistics["freight_per_km"] = (
    logistics["freight_value"]
    / logistics["real_distance_km"].replace(0, np.nan)
)

logistics["route_frequency"] = logistics.groupby(
    "state_pair"
)["order_id"].transform("count")

route_graph = (
    logistics.groupby(
        ["seller_state", "customer_state"]
    )
    .size()
    .reset_index(name="route_count")
)

G = nx.DiGraph()

for _, row in route_graph.iterrows():
    G.add_edge(
        row["seller_state"],
        row["customer_state"],
        weight=row["route_count"]
    )

corr_cols = [
    "estimated_delivery_days",
    "real_distance_km",
    "distance_squared",
    "log_distance",
    "distance_weight",
    "freight_value",
    "freight_per_km",
    "product_weight_g",
    "product_volume",
    "route_frequency",
    "same_state"
]

corr_matrix = logistics[
    corr_cols
].corr(numeric_only=True)

print("\n=== Correlation With estimated_delivery_days ===\n")

print(
    corr_matrix["estimated_delivery_days"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(16, 12))

mask = np.triu(
    np.ones_like(corr_matrix, dtype=bool)
)

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title(
    "Feature Correlation Matrix"
)

plt.show()

plt.figure(figsize=(12, 8))

plt.hexbin(
    logistics["real_distance_km"],
    logistics["estimated_delivery_days"],
    gridsize=70,
    cmap="viridis",
    bins="log"
)

plt.colorbar(label="log(count)")

plt.xlabel("Distance KM")
plt.ylabel("Estimated Delivery Days")

plt.title(
    "Density Map: Distance vs Delivery Days"
)

plt.show()

sample = logistics.sample(
    15000,
    random_state=42
)

smoothed = lowess(
    sample["estimated_delivery_days"],
    sample["real_distance_km"],
    frac=0.12
)

plt.figure(figsize=(12, 8))

plt.scatter(
    sample["real_distance_km"],
    sample["estimated_delivery_days"],
    alpha=0.05,
    s=10
)

plt.plot(
    smoothed[:, 0],
    smoothed[:, 1],
    linewidth=4
)

plt.xlabel("Distance KM")
plt.ylabel("Estimated Delivery Days")

plt.title(
    "LOWESS Nonlinear Distance Relationship"
)

plt.show()

plt.figure(figsize=(12, 8))

sns.boxplot(
    data=logistics,
    x=pd.qcut(
        logistics["real_distance_km"],
        10,
        duplicates="drop"
    ),
    y="estimated_delivery_days"
)

plt.xticks(rotation=45)

plt.xlabel("Distance Quantile Bucket")
plt.ylabel("Estimated Delivery Days")

plt.title(
    "Distance Regime Distribution"
)

plt.show()

plt.figure(figsize=(12, 8))

sns.violinplot(
    data=logistics,
    x="same_state",
    y="estimated_delivery_days"
)

plt.xlabel("Same State")
plt.ylabel("Estimated Delivery Days")

plt.title(
    "Same-State vs Interstate Delivery Regimes"
)

plt.show()

top_routes = route_graph.sort_values(
    by="route_count",
    ascending=False
).head(35)

G_small = nx.DiGraph()

for _, row in top_routes.iterrows():
    G_small.add_edge(
        row["seller_state"],
        row["customer_state"],
        weight=row["route_count"]
    )

plt.figure(figsize=(14, 12))

pos = nx.circular_layout(G_small)

weights = [
    G_small[u][v]["weight"] / 4000
    for u, v in G_small.edges()
]

nx.draw_networkx_nodes(
    G_small,
    pos,
    node_size=4500
)

nx.draw_networkx_labels(
    G_small,
    pos,
    font_size=11
)

nx.draw_networkx_edges(
    G_small,
    pos,
    width=weights,
    arrows=True,
    alpha=0.7
)

plt.title(
    "Top Brazilian Shipping Corridors"
)

plt.axis("off")

plt.show()

pivot = logistics.pivot_table(
    values="estimated_delivery_days",
    index="seller_state",
    columns="customer_state",
    aggfunc="mean"
)

plt.figure(figsize=(16, 12))

sns.heatmap(
    pivot,
    cmap="magma",
    linewidths=0.5
)

plt.title(
    "Average Delivery Days Between States"
)

plt.show()

top_categories = (
    logistics.groupby(
        "product_category_name"
    )["estimated_delivery_days"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(14, 8))

sns.barplot(
    x=top_categories.values,
    y=top_categories.index
)

plt.xlabel("Average Delivery Days")

plt.title(
    "Slowest Product Categories"
)

plt.show()

plt.figure(figsize=(12, 8))

sns.histplot(
    logistics["estimated_delivery_days"],
    bins=60,
    kde=True
)

plt.xlabel("Estimated Delivery Days")

plt.title(
    "Target Distribution"
)

plt.show()

plt.figure(figsize=(12, 8))

sns.kdeplot(
    data=sample,
    x="real_distance_km",
    y="estimated_delivery_days",
    fill=True,
    thresh=0.02
)

plt.xlabel("Distance KM")
plt.ylabel("Estimated Delivery Days")

plt.title(
    "2D Density Surface"
)

plt.show()