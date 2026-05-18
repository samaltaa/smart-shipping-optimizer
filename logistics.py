import sys
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
import seaborn as sns

from pathlib import Path
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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
print(logistics.info())

logistics["order_purchase_timestamp"] = pd.to_datetime(logistics["order_purchase_timestamp"])
logistics["order_delivered_customer_date"] = pd.to_datetime(logistics["order_delivered_customer_date"])
logistics["order_estimated_delivery_date"] = pd.to_datetime(logistics["order_estimated_delivery_date"])

logistics["late_delivery"] = (logistics["order_delivered_customer_date"] > logistics["order_estimated_delivery_date"]).astype(int)
logistics = logistics.dropna(subset=["late_delivery", "order_delivered_customer_date"])

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

logistics["price_cat"] = pd.cut(
    logistics["price"],
    bins=[-np.inf, 50, 100, 200, 500, np.inf],
    labels=[1, 2, 3, 4, 5]
)

logistics["seller_state_cat"] = logistics["seller_state"].astype("category").cat.codes
logistics["customer_state_cat"] = logistics["customer_state"].astype("category").cat.codes

cat_cols = ["freight_cat", "weight_cat", "price_cat", "seller_state_cat", "customer_state_cat"]
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

corr_matrix = logistics.corr(numeric_only=True)
print(corr_matrix["late_delivery"].sort_values(ascending=False))

attributes = [
    "freight_value",
    "price",
    "product_weight_g",
    "estimated_delivery_days",
    "late_delivery"
]

scatter_matrix(logistics[attributes], figsize=(12, 8))
#plt.show()

logistics.plot(kind="scatter", x="estimated_delivery_days", y="late_delivery",
               alpha=0.1, grid=True)
#plt.show()

logistics["price_per_gram"] = logistics["price"] / logistics["product_weight_g"].replace(0, np.nan)
logistics["freight_ratio"] = logistics["freight_value"] / logistics["price"].replace(0, np.nan)

corr_matrix = logistics.corr(numeric_only=True)
print(corr_matrix["late_delivery"].sort_values(ascending=False))

logistics = train_set.drop("late_delivery", axis=1)
logistics_labels = train_set["late_delivery"].copy()

logistics["order_purchase_timestamp"] = pd.to_datetime(logistics["order_purchase_timestamp"])
logistics["order_estimated_delivery_date"] = pd.to_datetime(logistics["order_estimated_delivery_date"])
logistics["estimated_delivery_days"] = (logistics["order_estimated_delivery_date"] - logistics["order_purchase_timestamp"]).dt.days
logistics["seller_state_cat"] = logistics["seller_state"].astype("category").cat.codes
logistics["customer_state_cat"] = logistics["customer_state"].astype("category").cat.codes
logistics["price_per_gram"] = logistics["price"] / logistics["product_weight_g"].replace(0, np.nan)
logistics["freight_ratio"] = logistics["freight_value"] / logistics["price"].replace(0, np.nan)

"""
Data cleaning
"""
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

"""
turn categoricals into numerical values
"""
logistics_categoricals = logistics[["seller_state", "customer_state", "product_category_name"]]
print(logistics_categoricals.head(8))

categorical_encoder = OneHotEncoder()
logistics_categorical_1hot = categorical_encoder.fit_transform(logistics_categoricals)
print(logistics_categorical_1hot.toarray())

df_test_unknown = pd.DataFrame({
    "seller_state": ["SP", "XX"],
    "customer_state": ["RJ", "YY"],
    "product_category_name": ["electronics", "unknown_category"]
})

categorical_encoder.handle_unknown = "ignore"
categorical_encoder.transform(df_test_unknown)

"""scaling and normalizing data"""
std_scaler = StandardScaler()
logistics_numerical_data_std_scaled = std_scaler.fit_transform(logistics_numerical_data_only)

"""
transform data
"""
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

logistics_num_prepared = num_pipeline.fit_transform(logistics_numerical_data_only)
print(logistics_num_prepared[:2].round(2))

num_attribs = ["freight_value", "price", "product_weight_g", "estimated_delivery_days",
               "seller_state_cat", "customer_state_cat", "price_per_gram", "freight_ratio"]
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

"""
training models
"""
logistic_regression = make_pipeline(make_preprocessing(), LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
logistic_regression.fit(logistics, logistics_labels)

tree_classifier = make_pipeline(make_preprocessing(), DecisionTreeClassifier(random_state=42, class_weight="balanced"))
tree_classifier.fit(logistics, logistics_labels)

forest_classifier = make_pipeline(make_preprocessing(), RandomForestClassifier(random_state=42, class_weight="balanced"))
forest_classifier.fit(logistics, logistics_labels)

print("Random Forest:")
forest_scores = cross_val_score(forest_classifier, logistics, logistics_labels,
                                scoring="f1", cv=10)
print(pd.Series(forest_scores).describe())

print("Decision Tree:")
tree_scores = cross_val_score(tree_classifier, logistics, logistics_labels,
                              scoring="f1", cv=10)
print(pd.Series(tree_scores).describe())

print("Logistic Regression:")
logistic_scores = cross_val_score(logistic_regression, logistics, logistics_labels,
                                  scoring="f1", cv=10)
print(pd.Series(logistic_scores).describe())