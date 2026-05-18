import sys
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
import seaborn as sns

from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint


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
print(logistics.info())

logistics["order_purchase_timestamp"] = pd.to_datetime(logistics["order_purchase_timestamp"])
logistics["order_delivered_customer_date"] = pd.to_datetime(logistics["order_delivered_customer_date"])
logistics["order_estimated_delivery_date"] = pd.to_datetime(logistics["order_estimated_delivery_date"])

logistics["estimated_delivery_days"] = (logistics["order_estimated_delivery_date"] - logistics["order_purchase_timestamp"]).dt.days
logistics = logistics.dropna(subset=["estimated_delivery_days", "order_delivered_customer_date"])

categoricals = ["order_status", "product_category_name", "seller_state", "customer_state"]
print(logistics[categoricals].value_counts())

train_set, test_set = train_test_split(logistics, test_size=0.2, random_state=42)
print(len(train_set), len(test_set))

logistics = logistics.dropna(subset=["product_weight_g", "freight_value", "price"])

logistics["freight_cat"] = pd.cut(
    logistics["freight_value"],
    bins=[-np.inf, 10, 20, 30, 50, np.inf],
    labels=[1, 2, 3, 4, 5]
)

logistics["weight_cat"] = pd.cut(
    logistics["product_weight_g"],
    bins=[-np.inf, 500, 1000, 2000, 5000, np.inf],
    labels=[1, 2, 3, 4, 5]
)

logistics["delivery_days_cat"] = pd.cut(
    logistics["estimated_delivery_days"],
    bins=[-np.inf, 10, 20, 30, 40, np.inf],
    labels=[1, 2, 3, 4, 5]
)

logistics["seller_state_cat"] = logistics["seller_state"].astype("category").cat.codes
logistics["customer_state_cat"] = logistics["customer_state"].astype("category").cat.codes

cat_cols = ["freight_cat", "weight_cat", "delivery_days_cat", "seller_state_cat", "customer_state_cat"]
print("\nNaN check:", logistics[cat_cols].isna().sum().to_dict())

def create_multiple_splits(n_splits, test_size, random_state, category):
    strat_train_set, strat_test_set = train_test_split(
        logistics, test_size=test_size, stratify=logistics[category], random_state=random_state
    )
    print(strat_test_set[category].value_counts() / len(strat_test_set))
    return strat_train_set, strat_test_set

splits = {}
for cat in cat_cols:
    print(f"\n=== {cat} ===")
    train, test = create_multiple_splits(
        n_splits=10, test_size=0.2, random_state=42, category=cat
    )
    splits[cat] = {"train": train, "test": test}

logistics = test_set.copy()

logistics["order_purchase_timestamp"] = pd.to_datetime(logistics["order_purchase_timestamp"])
logistics["order_estimated_delivery_date"] = pd.to_datetime(logistics["order_estimated_delivery_date"])
logistics["estimated_delivery_days"] = (logistics["order_estimated_delivery_date"] - logistics["order_purchase_timestamp"]).dt.days
logistics["seller_state_cat"] = logistics["seller_state"].astype("category").cat.codes
logistics["customer_state_cat"] = logistics["customer_state"].astype("category").cat.codes
logistics["zip_distance"] = abs(logistics["customer_zip_code_prefix"] - logistics["seller_zip_code_prefix"])
logistics["product_volume"] = logistics["product_length_cm"] * logistics["product_height_cm"] * logistics["product_width_cm"]
logistics["freight_ratio"] = logistics["freight_value"] / logistics["price"].replace(0, np.nan)
logistics["real_distance_km"] = logistics.apply(
    lambda row: haversine(row["seller_lat"], row["seller_lng"],
                          row["customer_lat"], row["customer_lng"]), axis=1
)
logistics["seller_customer_lat_diff"] = abs(logistics["customer_lat"] - logistics["seller_lat"])
logistics["seller_customer_lng_diff"] = abs(logistics["customer_lng"] - logistics["seller_lng"])
logistics["freight_per_km"] = logistics["freight_value"] / logistics["real_distance_km"].replace(0, np.nan)
logistics["price_per_km"] = logistics["price"] / logistics["real_distance_km"].replace(0, np.nan)

corr_matrix = logistics.corr(numeric_only=True)
print(corr_matrix["estimated_delivery_days"].sort_values(ascending=False))

attributes = [
    "real_distance_km",
    "seller_customer_lat_diff",
    "seller_customer_lng_diff",
    "freight_value",
    "estimated_delivery_days"
]

scatter_matrix(logistics[attributes], figsize=(12, 8))
#plt.show()

logistics.plot(kind="scatter", x="real_distance_km", y="estimated_delivery_days",
               alpha=0.1, grid=True)
#plt.show()

corr_matrix = logistics.corr(numeric_only=True)
print(corr_matrix["estimated_delivery_days"].sort_values(ascending=False))

logistics = train_set.drop("estimated_delivery_days", axis=1)
logistics_labels = train_set["estimated_delivery_days"].copy()

logistics["order_purchase_timestamp"] = pd.to_datetime(logistics["order_purchase_timestamp"])
logistics["order_estimated_delivery_date"] = pd.to_datetime(logistics["order_estimated_delivery_date"])
logistics["estimated_delivery_days"] = (logistics["order_estimated_delivery_date"] - logistics["order_purchase_timestamp"]).dt.days
logistics["seller_state_cat"] = logistics["seller_state"].astype("category").cat.codes
logistics["customer_state_cat"] = logistics["customer_state"].astype("category").cat.codes
logistics["zip_distance"] = abs(logistics["customer_zip_code_prefix"] - logistics["seller_zip_code_prefix"])
logistics["product_volume"] = logistics["product_length_cm"] * logistics["product_height_cm"] * logistics["product_width_cm"]
logistics["freight_ratio"] = logistics["freight_value"] / logistics["price"].replace(0, np.nan)
logistics["real_distance_km"] = logistics.apply(
    lambda row: haversine(row["seller_lat"], row["seller_lng"],
                          row["customer_lat"], row["customer_lng"]), axis=1
)
logistics["seller_customer_lat_diff"] = abs(logistics["customer_lat"] - logistics["seller_lat"])
logistics["seller_customer_lng_diff"] = abs(logistics["customer_lng"] - logistics["seller_lng"])
logistics["freight_per_km"] = logistics["freight_value"] / logistics["real_distance_km"].replace(0, np.nan)
logistics["price_per_km"] = logistics["price"] / logistics["real_distance_km"].replace(0, np.nan)

imputer = SimpleImputer(strategy="median")

logistics_numerical_data_only = logistics.select_dtypes(include=[np.number])
logistics_numerical_data_only = logistics_numerical_data_only.replace([np.inf, -np.inf], np.nan)

imputer.fit(logistics_numerical_data_only)
imputer.statistics_
print(logistics_numerical_data_only.median().values)

X = imputer.transform(logistics_numerical_data_only)

logistics_tr = pd.DataFrame(X,
                            columns=logistics_numerical_data_only.columns,
                            index=logistics_numerical_data_only.index)

logistics_categoricals = logistics[["seller_state", "customer_state", "product_category_name"]]
print(logistics_categoricals.head(8))

categorical_encoder = OneHotEncoder()
logistics_categorical_1hot = categorical_encoder.fit_transform(logistics_categoricals)
print(logistics_categorical_1hot.toarray())

df_test_unknown = pd.DataFrame({
    "seller_state": ["SP", "XX"],
    "customer_state": ["RJ", "YY"],
    "product_category_name": ["eletronicos", "unknown_category"]
})

categorical_encoder.handle_unknown = "ignore"
categorical_encoder.transform(df_test_unknown)

std_scaler = StandardScaler()
logistics_numerical_data_std_scaled = std_scaler.fit_transform(logistics_numerical_data_only)

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

logistics_num_prepared = num_pipeline.fit_transform(logistics_numerical_data_only)
print(logistics_num_prepared[:2].round(2))

num_attribs = ["real_distance_km", "seller_customer_lng_diff", "seller_customer_lat_diff",
               "zip_distance", "customer_zip_code_prefix", "customer_lat",
               "freight_value", "seller_zip_code_prefix", "freight_ratio",
               "product_weight_g", "product_volume", "customer_state_cat",
               "seller_state_cat", "freight_per_km", "price_per_km"]
cat_attribs = ["seller_state", "customer_state", "product_category_name"]

def make_preprocessing():
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )
    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

lin_reg = make_pipeline(make_preprocessing(), LinearRegression())
lin_reg.fit(logistics, logistics_labels)

tree_reg = make_pipeline(make_preprocessing(), DecisionTreeRegressor(random_state=42))
tree_reg.fit(logistics, logistics_labels)

forest_reg = make_pipeline(make_preprocessing(), RandomForestRegressor(random_state=42, n_estimators=50, max_depth=15, n_jobs=-1))
forest_reg.fit(logistics, logistics_labels)

print("Linear Regression:")
lin_scores = -cross_val_score(lin_reg, logistics, logistics_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(lin_scores).describe())

print("Decision Tree:")
tree_scores = -cross_val_score(tree_reg, logistics, logistics_labels,
                               scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(tree_scores).describe())

print("Random Forest:")
forest_scores = -cross_val_score(forest_reg, logistics, logistics_labels,
                                 scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(forest_scores).describe())