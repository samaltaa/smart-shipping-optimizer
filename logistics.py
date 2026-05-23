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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import randint
from xgboost import XGBRegressor


brazil_regions = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North",
    "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast",
    "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South"
}

category_complexity_map = {
    "moveis_escritorio": 5, "moveis_decoracao": 5, "moveis_quarto": 5,
    "moveis_sala": 5, "moveis_colchao_e_estofado": 5,
    "eletrodomesticos": 4, "eletrodomesticos_2": 4, "eletroportateis": 4,
    "informatica_acessorios": 3, "eletronicos": 3, "telefonia": 3,
    "esporte_lazer": 3, "brinquedos": 3, "ferramentas_jardim": 3,
    "beleza_saude": 2, "utilidades_domesticas": 2, "cama_mesa_banho": 2,
    "livros_tecnicos": 1, "livros_interesse_geral": 1, "musica": 1,
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def engineer_features(df, seller_volume_map, route_freq_map,
                      state_pair_avg_map, customer_state_avg_map,
                      seller_state_avg_map, route_variability_map,
                      seller_avg_review_map, seller_review_volatility_map,
                      seller_high_installment_rate_map, seller_avg_order_value_map,
                      seller_age_map, city_density_map, daily_order_map,
                      rolling_7d_map, product_first_sale_map,
                      is_train=True):
    df = df.copy()

    
    df["real_distance_km"] = df.apply(
        lambda row: haversine(row["seller_lat"], row["seller_lng"],
                              row["customer_lat"], row["customer_lng"]), axis=1
    )
    df["seller_customer_lat_diff"] = abs(df["customer_lat"] - df["seller_lat"])
    df["seller_customer_lng_diff"] = abs(df["customer_lng"] - df["seller_lng"])
    df["zip_distance"] = abs(df["customer_zip_code_prefix"] - df["seller_zip_code_prefix"])
    df["customer_zip_prefix_bin"] = df["customer_zip_code_prefix"] // 10000
    df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)
    df["state_pair"] = df["seller_state"] + "_" + df["customer_state"]
    df["state_pair_cat"] = df["state_pair"].astype("category").cat.codes
    df["seller_region"] = df["seller_state"].map(brazil_regions)
    df["customer_region"] = df["customer_state"].map(brazil_regions)
    df["seller_state_cat"] = df["seller_state"].astype("category").cat.codes
    df["customer_state_cat"] = df["customer_state"].astype("category").cat.codes
    df["north_involved"] = (
        (df["seller_region"] == "North") |
        (df["customer_region"] == "North")
    ).astype(int)
    df["remote_state_flag"] = (df["customer_region"] == "North").astype(int)
    df["seller_remote_flag"] = df["seller_state"].isin(
        ["AM", "RO", "PA", "AC", "RR", "AP", "TO"]
    ).astype(int)
    df["southeast_seller"] = (df["seller_region"] == "Southeast").astype(int)
    df["log_distance"] = np.log1p(df["real_distance_km"])
    df["extreme_longhaul_flag"] = (df["real_distance_km"] > 1450).astype(int)
    df["long_heavy"] = ((df["real_distance_km"] > 500) & (df["product_weight_g"] > 2000)).astype(int)
    df["short_heavy"] = ((df["real_distance_km"] < 200) & (df["product_weight_g"] > 5000)).astype(int)

   
    df["product_volume"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    df["freight_ratio"] = df["freight_value"] / df["price"].replace(0, np.nan)
    df["freight_per_km"] = df["freight_value"] / df["real_distance_km"].replace(0, np.nan)
    df["price_per_km"] = df["price"] / df["real_distance_km"].replace(0, np.nan)
    df["freight_per_weight"] = df["freight_value"] / df["product_weight_g"].replace(0, np.nan)
    df["max_dimension_cm"] = df[["product_length_cm", "product_height_cm", "product_width_cm"]].max(axis=1)
    df["log_freight"] = np.log1p(df["freight_value"])
    df["log_weight"] = np.log1p(df["product_weight_g"])
    df["log_volume"] = np.log1p(df["product_volume"])
    df["log_total_weight"] = np.log1p(df["product_weight_g"])
    df["heavy_item_flag"] = (df["product_weight_g"] > 5000).astype(int)
    df["local_heavy"] = ((df["same_state"] == 1) & (df["product_weight_g"] > 5000)).astype(int)
    df["ultra_cheap_freight"] = (df["freight_value"] < 10).astype(int)
    df["freight_tier"] = pd.cut(
        df["freight_value"],
        bins=[0, 10, 20, 50, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(float)

    
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["purchase_date"] = df["order_purchase_timestamp"].dt.date
    df["holiday_pressure"] = df["purchase_month"].isin([1, 2, 6, 12]).astype(int)
    df["fast_season"] = df["purchase_month"].isin([7, 8, 9]).astype(int)
    df["daily_order_count"] = df["purchase_date"].map(daily_order_map).fillna(0)
    df["rolling_7d_orders"] = df["purchase_date"].map(rolling_7d_map).fillna(0)
    df["log_rolling_7d"] = np.log1p(df["rolling_7d_orders"])

    
    if is_train:
        df["seller_order_volume"] = df.groupby("seller_id")["order_id"].transform("count")
        df["route_frequency"] = df.groupby("state_pair")["order_id"].transform("count")
    else:
        df["seller_order_volume"] = df["seller_id"].map(seller_volume_map).fillna(0)
        df["route_frequency"] = df["state_pair"].map(route_freq_map).fillna(0)

    df["log_distance_route_ratio"] = np.log1p(
        df["real_distance_km"] / df["route_frequency"].replace(0, np.nan)
    )
    df["rare_route_flag"] = (df["route_frequency"] < df["route_frequency"].quantile(0.25)).astype(int)
    df["route_variability"] = df["state_pair"].map(route_variability_map).fillna(route_variability_map.mean())
    df["state_pair_avg_days"] = df["state_pair"].map(state_pair_avg_map).fillna(state_pair_avg_map.mean())
    df["customer_state_avg_days"] = df["customer_state"].map(customer_state_avg_map).fillna(customer_state_avg_map.mean())
    df["seller_state_avg_days"] = df["seller_state"].map(seller_state_avg_map).fillna(seller_state_avg_map.mean())

    
    df["shipping_limit_days"] = (
        df["shipping_limit_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["seller_processing_window"] = (
        df["shipping_limit_date"] - df["order_approved_at"]
    ).dt.total_seconds() / 3600
    df["payment_approval_delay"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600
    df["log_approval_delay"] = np.log1p(
        df["payment_approval_delay"].replace([np.inf, -np.inf], np.nan)
    )

    
    df["category_complexity"] = df["product_category_name"].map(category_complexity_map).fillna(2)

    
    order_agg = df.groupby("order_id").agg(
        items_per_order=("order_item_id", "count"),
        unique_sellers_per_order=("seller_id", "nunique"),
        total_weight=("product_weight_g", "sum"),
        total_volume=("product_volume", "sum"),
        order_total_price=("price", "sum")
    )
    df = df.join(order_agg, on="order_id", rsuffix="_agg")
    df["log_total_order_weight"] = np.log1p(df["total_weight"])
    df["log_installments"] = np.log1p(df["payment_installments"])

    #seller maps
    df["seller_avg_review"] = df["seller_id"].map(seller_avg_review_map).fillna(3.0)
    df["seller_review_volatility"] = df["seller_id"].map(seller_review_volatility_map).fillna(0)
    df["seller_high_installment_rate"] = df["seller_id"].map(seller_high_installment_rate_map).fillna(0)
    df["seller_avg_order_value"] = df["seller_id"].map(seller_avg_order_value_map).fillna(
        seller_avg_order_value_map.mean()
    )
    df["seller_age_days"] = df["seller_id"].map(seller_age_map).fillna(0)
    df["log_seller_age"] = np.log1p(df["seller_age_days"])

    #customer and city combinations
    df["customer_city_order_density"] = df["customer_city"].map(city_density_map).fillna(0)
    df["log_city_density"] = np.log1p(df["customer_city_order_density"])
    df["log_clv"] = np.log1p(df["price"])

    #payment features
    df["payment_value_vs_order_value"] = (
        df["payment_value"] / df["order_total_price"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    df["high_complexity_installment"] = df["category_complexity"] * df["payment_installments"]
    df["installment_approval_lag"] = (
        df["payment_installments"] *
        df["log_approval_delay"].replace([np.inf, -np.inf], np.nan)
    )

    #product cat age
    df["product_first_sale"] = df["product_id"].map(product_first_sale_map)
    df["product_catalog_age_days"] = (
        df["order_purchase_timestamp"] - df["product_first_sale"]
    ).dt.days
    df["log_catalog_age"] = np.log1p(df["product_catalog_age_days"].fillna(0))
    df["log_review_comment"] = np.log1p(df["review_comment_length"].fillna(0))
    df["log_order_value_ratio"] = np.log1p(
        df["price"] / df["seller_avg_order_value"].replace(0, np.nan)
    )

    #operational stress composites
    df["freight_burden"] = (
        df["freight_ratio"].replace([np.inf, -np.inf], np.nan) *
        df["log_total_order_weight"]
    )
    df["complex_heavy_order"] = df["category_complexity"] * df["log_total_order_weight"]
    df["clv_x_complexity"] = df["log_clv"] * df["category_complexity"]
    df["weight_x_sellers"] = df["log_total_order_weight"] * df["unique_sellers_per_order"]
    df["operational_stress"] = (
        df["log_approval_delay"].fillna(0) +
        df["unique_sellers_per_order"].fillna(1) +
        df["log_installments"].fillna(0)
    )
    df["pressure_x_weight_sellers"] = (
        df["rolling_7d_orders"] * df["weight_x_sellers"]
    ).replace([np.inf, -np.inf], np.nan)
    df["pressure_x_operational_stress"] = (
        df["daily_order_count"] * df["operational_stress"]
    ).replace([np.inf, -np.inf], np.nan)

    #shipping window combinations
    df["shipping_window_x_complexity"] = df["shipping_limit_days"] * df["category_complexity"]
    df["shipping_window_x_clv"] = df["shipping_window_x_complexity"] * df["log_clv"]
    df["shipping_window_x_seller_age"] = df["shipping_limit_days"] * df["log_seller_age"]
    df["shipping_window_x_operational_stress"] = df["shipping_limit_days"] * df["operational_stress"]
    df["shipping_window_x_city_density"] = df["shipping_window_x_complexity"] * df["log_city_density"]
    df["shipping_window_x_installment_lag"] = (
        df["shipping_limit_days"] *
        df["installment_approval_lag"].replace([np.inf, -np.inf], np.nan)
    )
    df["high_installment_x_shipping_window"] = df["high_complexity_installment"] * df["shipping_limit_days"]
    df["load_x_shipping_window"] = df["log_rolling_7d"] * df["shipping_limit_days"]

    #processing window combinations
    df["processing_window_x_complexity"] = (
        df["seller_processing_window"] * df["category_complexity"]
    ).replace([np.inf, -np.inf], np.nan)
    df["processing_window_x_clv_complexity"] = (
        df["seller_processing_window"] * df["clv_x_complexity"]
    ).replace([np.inf, -np.inf], np.nan)
    df["processing_window_x_seller_age"] = (
        df["seller_processing_window"] * df["log_seller_age"]
    ).replace([np.inf, -np.inf], np.nan)
    df["processing_window_x_weight"] = (
        df["seller_processing_window"] * df["log_total_order_weight"]
    ).replace([np.inf, -np.inf], np.nan)
    df["processing_window_x_installments"] = (
        df["seller_processing_window"] * df["log_installments"]
    ).replace([np.inf, -np.inf], np.nan)
    df["payment_complexity_x_processing"] = (
        df["payment_value_vs_order_value"].replace([np.inf, -np.inf], np.nan) *
        df["seller_processing_window"]
    ).replace([np.inf, -np.inf], np.nan)

    #city density combinations
    df["city_density_x_seller_age"] = df["log_city_density"] * df["log_seller_age"]
    df["city_density_x_clv"] = df["log_city_density"] * df["log_clv"]
    df["city_density_x_weight_sellers"] = df["log_city_density"] * df["weight_x_sellers"]
    df["city_density_x_operational_stress"] = df["log_city_density"] * df["operational_stress"]

    return df


def load_data():
    path = Path("C:/Users/Grace/mlprojects/data/")
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
        payment_sequential=("payment_sequential", "max")
    ).reset_index()
    df = df.merge(payments_agg, on="order_id", how="left")
    reviews_agg = order_reviews.groupby("order_id").agg(
        review_score=("review_score", "mean"),
        review_comment_length=("review_comment_message", lambda x: x.dropna().str.len().mean())
    ).reset_index()
    df = df.merge(reviews_agg, on="order_id", how="left")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_approved_at"] = pd.to_datetime(df["order_approved_at"])
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    return df


logistics = load_data()
print(logistics.info())

logistics["estimated_delivery_days"] = (
    logistics["order_estimated_delivery_date"] - logistics["order_purchase_timestamp"]
).dt.days
logistics = logistics.dropna(subset=["estimated_delivery_days", "order_delivered_customer_date"])

train_set, test_set = train_test_split(logistics, test_size=0.2, random_state=42)
print(len(train_set), len(test_set))

train_set = train_set.dropna(subset=["product_weight_g", "freight_value", "price"])
train_set = train_set.copy()
train_set["estimated_delivery_days"] = (
    train_set["order_estimated_delivery_date"] - train_set["order_purchase_timestamp"]
).dt.days
train_set["state_pair"] = train_set["seller_state"] + "_" + train_set["customer_state"]
train_set["purchase_date"] = train_set["order_purchase_timestamp"].dt.date

seller_volume_map = train_set.groupby("seller_id")["order_id"].count()
route_freq_map = train_set.groupby("state_pair")["order_id"].count()
state_pair_avg_map = train_set.groupby("state_pair")["estimated_delivery_days"].mean()
customer_state_avg_map = train_set.groupby("customer_state")["estimated_delivery_days"].mean()
seller_state_avg_map = train_set.groupby("seller_state")["estimated_delivery_days"].mean()
route_variability_map = train_set.groupby("state_pair")["estimated_delivery_days"].std()
seller_avg_review_map = train_set.groupby("seller_id")["review_score"].mean()
seller_review_volatility_map = train_set.groupby("seller_id")["review_score"].std().fillna(0)
seller_high_installment_map = train_set.groupby("seller_id")["payment_installments"].apply(
    lambda x: (x > 3).mean()
)
seller_avg_order_value_map = train_set.groupby("seller_id")["price"].mean()
seller_first_sale_map = train_set.groupby("seller_id")["order_purchase_timestamp"].min()
seller_age_map = (train_set["order_purchase_timestamp"].max() - seller_first_sale_map).dt.days
city_density_map = train_set.groupby("customer_city")["order_id"].count()
product_first_sale_map = train_set.groupby("product_id")["order_purchase_timestamp"].min()

daily_orders = train_set.groupby("purchase_date").size().reset_index(name="daily_order_count")
daily_orders["purchase_date"] = pd.to_datetime(daily_orders["purchase_date"])
daily_orders = daily_orders.sort_values("purchase_date")
daily_orders["rolling_7d_orders"] = daily_orders["daily_order_count"].rolling(7).mean()
daily_order_map = daily_orders.set_index("purchase_date")["daily_order_count"].to_dict()
rolling_7d_map = daily_orders.set_index("purchase_date")["rolling_7d_orders"].to_dict()

logistics_for_strat = train_set.copy()

logistics_for_strat["freight_cat"] = pd.cut(
    logistics_for_strat["freight_value"],
    bins=[-np.inf, 10, 20, 30, 50, np.inf],
    labels=[1, 2, 3, 4, 5]
)
logistics_for_strat["weight_cat"] = pd.cut(
    logistics_for_strat["product_weight_g"],
    bins=[-np.inf, 500, 1000, 2000, 5000, np.inf],
    labels=[1, 2, 3, 4, 5]
)
logistics_for_strat["delivery_days_cat"] = pd.cut(
    logistics_for_strat["estimated_delivery_days"],
    bins=[-np.inf, 10, 20, 30, 40, np.inf],
    labels=[1, 2, 3, 4, 5]
)
logistics_for_strat["seller_state_cat"] = logistics_for_strat["seller_state"].astype("category").cat.codes
logistics_for_strat["customer_state_cat"] = logistics_for_strat["customer_state"].astype("category").cat.codes

cat_cols = ["freight_cat", "weight_cat", "delivery_days_cat", "seller_state_cat", "customer_state_cat"]
print("\nNaN check:", logistics_for_strat[cat_cols].isna().sum().to_dict())

categoricals = ["order_status", "product_category_name", "seller_state", "customer_state"]
print(logistics_for_strat[categoricals].value_counts())


def create_multiple_splits(n_splits, test_size, random_state, category):
    strat_train_set, strat_test_set = train_test_split(
        logistics_for_strat, test_size=test_size,
        stratify=logistics_for_strat[category], random_state=random_state
    )
    print(strat_test_set[category].value_counts() / len(strat_test_set))
    return strat_train_set, strat_test_set


splits = {}
for cat in cat_cols:
    print(f"\n=== {cat} ===")
    strat_train, strat_test = create_multiple_splits(
        n_splits=10, test_size=0.2, random_state=42, category=cat
    )
    splits[cat] = {"train": strat_train, "test": strat_test}

exploration = test_set.copy()
exploration = exploration.dropna(subset=["product_weight_g", "freight_value", "price"])
exploration["estimated_delivery_days"] = (
    exploration["order_estimated_delivery_date"] - exploration["order_purchase_timestamp"]
).dt.days
exploration = engineer_features(
    exploration, seller_volume_map, route_freq_map,
    state_pair_avg_map, customer_state_avg_map, seller_state_avg_map,
    route_variability_map, seller_avg_review_map, seller_review_volatility_map,
    seller_high_installment_map, seller_avg_order_value_map, seller_age_map,
    city_density_map, daily_order_map, rolling_7d_map, product_first_sale_map,
    is_train=False
)

corr_matrix = exploration.corr(numeric_only=True)
print(corr_matrix["estimated_delivery_days"].sort_values(ascending=False))

attributes = [
    "real_distance_km",
    "seller_customer_lat_diff",
    "seller_customer_lng_diff",
    "freight_value",
    "estimated_delivery_days"
]
scatter_matrix(exploration[attributes], figsize=(12, 8))
exploration.plot(kind="scatter", x="real_distance_km", y="estimated_delivery_days",
                 alpha=0.1, grid=True)

logistics = train_set.drop("estimated_delivery_days", axis=1)
logistics_labels = train_set["estimated_delivery_days"].copy()

logistics = engineer_features(
    logistics, seller_volume_map, route_freq_map,
    state_pair_avg_map, customer_state_avg_map, seller_state_avg_map,
    route_variability_map, seller_avg_review_map, seller_review_volatility_map,
    seller_high_installment_map, seller_avg_order_value_map, seller_age_map,
    city_density_map, daily_order_map, rolling_7d_map, product_first_sale_map,
    is_train=True
)

imputer = SimpleImputer(strategy="median")
logistics_numerical_data_only = logistics.select_dtypes(include=[np.number])
logistics_numerical_data_only = logistics_numerical_data_only.replace([np.inf, -np.inf], np.nan)
imputer.fit(logistics_numerical_data_only)
print(logistics_numerical_data_only.median().values)

X = imputer.transform(logistics_numerical_data_only)
logistics_tr = pd.DataFrame(X,
                            columns=logistics_numerical_data_only.columns,
                            index=logistics_numerical_data_only.index)

logistics_categoricals = logistics[["seller_state", "customer_state", "product_category_name",
                                    "seller_region", "customer_region"]]
print(logistics_categoricals.head(8))

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
logistics_categorical_1hot = categorical_encoder.fit_transform(logistics_categoricals)

df_test_unknown = pd.DataFrame({
    "seller_state": ["SP", "XX"],
    "customer_state": ["RJ", "YY"],
    "product_category_name": ["eletronicos", "unknown_category"],
    "seller_region": ["Southeast", "Unknown"],
    "customer_region": ["Southeast", "Unknown"]
})
categorical_encoder.transform(df_test_unknown)

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
logistics_num_prepared = num_pipeline.fit_transform(logistics_numerical_data_only)
print(logistics_num_prepared[:2].round(2))

num_attribs = [
    # geographic
    "real_distance_km", "log_distance", "seller_customer_lng_diff",
    "seller_customer_lat_diff", "zip_distance", "customer_zip_code_prefix",
    "customer_zip_prefix_bin", "customer_lat", "seller_zip_code_prefix",
    "same_state", "north_involved", "remote_state_flag", "seller_remote_flag",
    "southeast_seller", "extreme_longhaul_flag", "long_heavy", "short_heavy",
    "local_heavy", "heavy_item_flag", "log_distance_route_ratio",
    # freight / product
    "freight_value", "log_freight", "freight_tier", "ultra_cheap_freight",
    "freight_ratio", "freight_per_km", "freight_per_weight",
    "product_weight_g", "log_weight", "product_volume", "log_volume",
    "max_dimension_cm", "price_per_km",
    # temporal
    "purchase_month", "purchase_dayofweek", "quarter",
    "holiday_pressure", "fast_season",
    "daily_order_count", "rolling_7d_orders", "log_rolling_7d",
    # route
    "route_frequency", "rare_route_flag", "route_variability",
    "state_pair_cat", "state_pair_avg_days",
    "customer_state_avg_days", "seller_state_avg_days",
    "customer_state_cat", "seller_state_cat",
    # shipping window
    "shipping_limit_days", "seller_processing_window",
    "payment_approval_delay", "log_approval_delay",
    # category / product
    "category_complexity", "log_catalog_age",
    # order aggregates
    "items_per_order", "unique_sellers_per_order",
    "total_weight", "log_total_order_weight",
    # payment
    "payment_installments", "log_installments",
    "payment_value_vs_order_value",
    "high_complexity_installment", "installment_approval_lag",
    # seller
    "seller_avg_review", "seller_review_volatility",
    "seller_high_installment_rate", "seller_age_days", "log_seller_age",
    "log_order_value_ratio", "log_review_comment",
    # customer / city
    "customer_city_order_density", "log_city_density", "log_clv",
    # operational stress composites
    "freight_burden", "complex_heavy_order", "clv_x_complexity",
    "weight_x_sellers", "operational_stress",
    "pressure_x_weight_sellers", "pressure_x_operational_stress",
    # shipping window combinations
    "shipping_window_x_complexity", "shipping_window_x_clv",
    "shipping_window_x_seller_age", "shipping_window_x_operational_stress",
    "shipping_window_x_city_density", "shipping_window_x_installment_lag",
    "high_installment_x_shipping_window", "load_x_shipping_window",
    # processing window combinations
    "processing_window_x_complexity", "processing_window_x_clv_complexity",
    "processing_window_x_seller_age", "processing_window_x_weight",
    "processing_window_x_installments", "payment_complexity_x_processing",
    # city density combinations
    "city_density_x_seller_age", "city_density_x_clv",
    "city_density_x_weight_sellers", "city_density_x_operational_stress",
]

cat_attribs = ["seller_state", "customer_state", "product_category_name",
               "seller_region", "customer_region"]

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

gradient_boost_reg = make_pipeline(make_preprocessing(), GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
))
gradient_boost_reg.fit(logistics, logistics_labels)

print("Gradient Boosting:")
gb_scores = -cross_val_score(gradient_boost_reg, logistics, logistics_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(gb_scores).describe())

print("XGBoost baseline:")
xgb_reg = make_pipeline(make_preprocessing(), XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
))
xgb_reg.fit(logistics, logistics_labels)
xgb_scores = -cross_val_score(xgb_reg, logistics, logistics_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(xgb_scores).describe())

xgb_param_distribs = {
    "xgbregressor__n_estimators": randint(300, 1000),
    "xgbregressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "xgbregressor__max_depth": randint(4, 10),
    "xgbregressor__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "xgbregressor__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "xgbregressor__min_child_weight": randint(1, 10),
    "xgbregressor__gamma": [0, 0.1, 0.2, 0.5],
    "xgbregressor__reg_alpha": [0, 0.1, 0.5, 1.0],
    "xgbregressor__reg_lambda": [0.5, 1.0, 1.5, 2.0]
}

xgb_search = RandomizedSearchCV(
    xgb_reg,
    xgb_param_distribs,
    n_iter=100,
    cv=10,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=3,
    return_train_score=True,
    error_score="raise"
)

xgb_search.fit(logistics, logistics_labels)

print(f"\nXGBoost Best RMSE: {-xgb_search.best_score_:.4f} days")
print(f"\nXGBoost Best params:\n{xgb_search.best_params_}")

xgb_cv_results = pd.DataFrame(xgb_search.cv_results_)
print("\nTop 10 XGBoost combinations:")
print(xgb_cv_results[[
    "param_xgbregressor__n_estimators",
    "param_xgbregressor__learning_rate",
    "param_xgbregressor__max_depth",
    "param_xgbregressor__subsample",
    "param_xgbregressor__colsample_bytree",
    "mean_test_score",
    "mean_train_score",
    "std_test_score"
]].sort_values("mean_test_score", ascending=False).head(10).to_string())