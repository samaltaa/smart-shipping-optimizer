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
    return pd.read_csv(Path("C:/Users/Grace/mlprojects/logistics.csv"), encoding="latin-1")

logistics = load_data()
print(logistics.info())

categoricals = ["Type", "Delivery Status", "Customer Segment", "Shipping Mode", "Market"]
print(logistics[categoricals].value_counts())

train_set, test_set = train_test_split(logistics, test_size=0.2, random_state=42)
print(len(train_set), len(test_set))

logistics["shipping_mode_cat"] = logistics["Shipping Mode"].astype("category").cat.codes
logistics["market_cat"] = logistics["Market"].astype("category").cat.codes

logistics["shipping_days_cat"] = pd.cut(
    logistics["Days for shipping (real)"],
    bins=[-np.inf, 1, 2, 3, 4, np.inf],
    labels=[1, 2, 3, 4, 5]
)

logistics["benefit_cat"] = pd.cut(
    logistics["Benefit per order"],
    bins=[-np.inf, 0, 50, 100, 200, np.inf],
    labels=[1, 2, 3, 4, 5]
)

logistics["sales_cat"] = pd.cut(
    logistics["Sales per customer"],
    bins=[-np.inf, 50, 100, 200, 300, np.inf],
    labels=[1, 2, 3, 4, 5]
)

logistics["discount_cat"] = pd.cut(
    logistics["Order Item Discount Rate"],
    bins=[-np.inf, 0.05, 0.10, 0.15, 0.20, np.inf],
    labels=[1, 2, 3, 4, 5]
)


cat_cols = ["shipping_days_cat", "benefit_cat", "sales_cat", "discount_cat", "shipping_mode_cat", "market_cat"]
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

#copy test set to keep using
logistics = test_set.copy()

logistics["shipping_mode_cat"] = logistics["Shipping Mode"].astype("category").cat.codes

corr_matrix = logistics.corr(numeric_only=True)
print(corr_matrix["Late_delivery_risk"].sort_values(ascending=False))

attributes = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Late_delivery_risk"
]

scatter_matrix(logistics[attributes], figsize=(12, 8))
#plt.show()

logistics.plot(kind="scatter", x="Days for shipping (real)", y="Late_delivery_risk",
               alpha=0.1, grid=True)
#plt.show()


corr_matrix = logistics.corr(numeric_only=True)
print(corr_matrix["Late_delivery_risk"].sort_values(ascending=False))

#drop target
logistics = train_set.drop("Late_delivery_risk", axis=1)
logistics_labels = train_set["Late_delivery_risk"].copy()
logistics = logistics.drop("Product Description", axis=1)

#feature engineering 
logistics["shipping_efficiency"] = logistics["Days for shipment (scheduled)"] / logistics["Days for shipping (real)"]
logistics["delay_delta"] = logistics["Days for shipping (real)"] - logistics["Days for shipment (scheduled)"]
logistics["shipping_mode_cat"] = logistics["Shipping Mode"].astype("category").cat.codes

"""
Data cleaning 
"""
#imputing missing values by replacing them with the median 
imputer = SimpleImputer(strategy="median")

#drop categoricals to only compute numerical data
logistics_numerical_data_only = logistics.select_dtypes(include=[np.number])

#fit imputer to the data and impute median to all new data too
imputer.fit(logistics_numerical_data_only)
imputer.statistics_
print(logistics_numerical_data_only.median().values)

#transform the training set with replaced missing values
X = imputer.transform(logistics_numerical_data_only)

#convert transformed data back into dataframe 
logistics_tr = pd.DataFrame(X,
                            columns=logistics_numerical_data_only.columns,
                            index=logistics_numerical_data_only.index)

"""
turn categoricals into numerical values
"""
logistics_categoticals = logistics[["Shipping Mode"]]
print(logistics_categoticals.head(8))

#one-hot encoding
categorical_encoder = OneHotEncoder()
logistics_categorical_1hot = categorical_encoder.fit_transform(logistics_categoticals)
print(logistics_categorical_1hot.toarray())

#handle unknown values: test by using a value that exists (same day) and one that doesn't (express)
df_test_unknown = pd.DataFrame({"Shipping Mode": ["Same Day", "Express"]})

#set unknown handler param to "ignore"
categorical_encoder.handle_unknown = "ignore"
categorical_encoder.transform(df_test_unknown)


"""scaling and normalizing data"""
#scale and normalize data to account for outliers
std_scaler = StandardScaler()
logistics_numerical_data_std_scaled = std_scaler.fit_transform(logistics_numerical_data_only)

"""
transform data
"""
#ColumnTransform() to handle categoricals and numericals at once
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

logistics_num_prepared = num_pipeline.fit_transform(logistics_numerical_data_only)
print(logistics_num_prepared[:2].round(2))

num_attribs = ["Days for shipping (real)", "Days for shipment (scheduled)",
                "shipping_efficiency", "delay_delta", "shipping_mode_cat"]
cat_attribs = ["Shipping Mode"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

"""
training models
"""
#logistic regression model
logistic_regression = make_pipeline(preprocessing, LogisticRegression(random_state=42, max_iter=1000))
logistic_regression.fit(logistics, logistics_labels)

#decision tree classifier model
tree_classifier = make_pipeline(preprocessing, DecisionTreeClassifier(random_state=42))
tree_classifier.fit(logistics, logistics_labels)

#random forest classifier model
forest_classifier = make_pipeline(preprocessing, RandomForestClassifier(random_state=42))
forest_classifier.fit(logistics, logistics_labels)

forest_scores = cross_val_score(forest_classifier, logistics, logistics_labels,
                                 scoring="f1", cv=10)
print(pd.Series(forest_scores).describe())

tree_scores = cross_val_score(tree_classifier, logistics, logistics_labels,
                              scoring="f1", cv=10)
print(pd.Series(tree_scores).describe())

logistic_scores = cross_val_score(logistic_regression, logistics, logistics_labels,
                                  scoring="f1", cv=10)
print(pd.Series(logistic_scores).describe())