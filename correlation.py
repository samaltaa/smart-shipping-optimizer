from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    path = Path("C:/Users/Grace/mlprojects/data/")
    orders = pd.read_csv(path / "olist_orders_dataset.csv")
    order_items = pd.read_csv(path / "olist_order_items_dataset.csv")
    products = pd.read_csv(path / "olist_products_dataset.csv")
    sellers = pd.read_csv(path / "olist_sellers_dataset.csv")
    customers = pd.read_csv(path / "olist_customers_dataset.csv")
    df = orders.merge(order_items, on="order_id")
    df = df.merge(products, on="product_id")
    df = df.merge(sellers, on="seller_id")
    df = df.merge(customers, on="customer_id")
    return df

logistics = load_data()

logistics["order_purchase_timestamp"] = pd.to_datetime(logistics["order_purchase_timestamp"])
logistics["order_delivered_customer_date"] = pd.to_datetime(logistics["order_delivered_customer_date"])
logistics["order_estimated_delivery_date"] = pd.to_datetime(logistics["order_estimated_delivery_date"])

logistics["late_delivery"] = (logistics["order_delivered_customer_date"] > logistics["order_estimated_delivery_date"]).astype(int)
logistics = logistics.dropna(subset=["late_delivery", "order_delivered_customer_date"])

logistics["estimated_delivery_days"] = (logistics["order_estimated_delivery_date"] - logistics["order_purchase_timestamp"]).dt.days
logistics["seller_state_cat"] = logistics["seller_state"].astype("category").cat.codes
logistics["customer_state_cat"] = logistics["customer_state"].astype("category").cat.codes
logistics["price_per_gram"] = logistics["price"] / logistics["product_weight_g"].replace(0, np.nan)
logistics["freight_ratio"] = logistics["freight_value"] / logistics["price"].replace(0, np.nan)
logistics["zip_distance"] = abs(logistics["customer_zip_code_prefix"] - logistics["seller_zip_code_prefix"])
logistics["product_volume"] = logistics["product_length_cm"] * logistics["product_height_cm"] * logistics["product_width_cm"]
logistics["product_density"] = logistics["product_weight_g"] / logistics["product_volume"].replace(0, np.nan)
logistics["weight_freight_ratio"] = logistics["product_weight_g"] / logistics["freight_value"].replace(0, np.nan)

corr_matrix = logistics.corr(numeric_only=True)
print(corr_matrix["late_delivery"].sort_values(ascending=False))

print("\n=== Categorical feature late delivery rates ===")
for col in ["order_status", "product_category_name", "seller_state", "customer_state", "seller_city", "customer_city"]:
    print(f"\n{col}:")
    print(logistics.groupby(col)["late_delivery"].mean().sort_values(ascending=False).head(10))

print(logistics["estimated_delivery_days"].describe())
logistics["estimated_delivery_days"].hist(bins=50, grid=True)
plt.xlabel("estimated_delivery_days")
plt.ylabel("count")
plt.show()

corr_matrix = logistics.corr(numeric_only=True)
print(corr_matrix["estimated_delivery_days"].sort_values(ascending=False))

for col in ["seller_state", "customer_state", "product_category_name"]:
    print(f"\n{col}:")
    print(logistics.groupby(col)["estimated_delivery_days"].mean().sort_values(ascending=False).head(10))