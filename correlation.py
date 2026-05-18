from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def load_data():
    path = Path("C:/Users/Grace/mlprojects/data/")
    orders = pd.read_csv(path / "olist_orders_dataset.csv")
    order_items = pd.read_csv(path / "olist_order_items_dataset.csv")
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
    return df

logistics = load_data()

logistics["order_purchase_timestamp"] = pd.to_datetime(logistics["order_purchase_timestamp"])
logistics["order_delivered_customer_date"] = pd.to_datetime(logistics["order_delivered_customer_date"])
logistics["order_estimated_delivery_date"] = pd.to_datetime(logistics["order_estimated_delivery_date"])

logistics["estimated_delivery_days"] = (logistics["order_estimated_delivery_date"] - logistics["order_purchase_timestamp"]).dt.days
logistics = logistics.dropna(subset=["estimated_delivery_days", "order_delivered_customer_date"])

logistics["seller_state_cat"] = logistics["seller_state"].astype("category").cat.codes
logistics["customer_state_cat"] = logistics["customer_state"].astype("category").cat.codes
logistics["zip_distance"] = abs(logistics["customer_zip_code_prefix"] - logistics["seller_zip_code_prefix"])
logistics["product_volume"] = logistics["product_length_cm"] * logistics["product_height_cm"] * logistics["product_width_cm"]
logistics["freight_ratio"] = logistics["freight_value"] / logistics["price"].replace(0, np.nan)
logistics["real_distance_km"] = logistics.apply(
    lambda row: haversine(row["seller_lat"], row["seller_lng"],
                          row["customer_lat"], row["customer_lng"]), axis=1
)

corr_matrix = logistics.corr(numeric_only=True)
print(corr_matrix["estimated_delivery_days"].sort_values(ascending=False))

print("\n=== Categorical feature delivery day averages ===")
for col in ["seller_state", "customer_state", "product_category_name"]:
    print(f"\n{col}:")
    print(logistics.groupby(col)["estimated_delivery_days"].mean().sort_values(ascending=False).head(10))

print(logistics["estimated_delivery_days"].describe())
logistics["estimated_delivery_days"].hist(bins=50, grid=True)
plt.xlabel("estimated_delivery_days")
plt.ylabel("count")
plt.show()