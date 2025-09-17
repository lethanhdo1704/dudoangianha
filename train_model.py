import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib

# ================== Load data ==================
FILE_PATH = "data/vietnam_housing_dataset.csv"
LOCATIONS_FILE = "locations.json"
CHOICES_FILE = "choices.json"

df = pd.read_csv(FILE_PATH)

# ================== Split Address ==================
def split_address(addr: str):
    if pd.isnull(addr) or str(addr).strip() == "":
        return pd.Series({"City": "N/A", "Province": "Khác"})
    parts = [p.strip() for p in str(addr).split(",") if p.strip()]
    province = parts[-1] if len(parts) >= 1 else "Khác"
    city = parts[-2] if len(parts) >= 2 else "N/A"
    return pd.Series({"City": city, "Province": province})

df[["City", "Province"]] = df["Address"].apply(split_address)

# ================== Normalize Province ==================
province_mapping = {
    "HN": "Hà Nội", "Hà Nội": "Hà Nội", "Hà Nội.": "Hà Nội",
    "TP Hồ Chí Minh": "Hồ Chí Minh", "TP. HCM": "Hồ Chí Minh",
    "TPHCM": "Hồ Chí Minh", "TpHCM": "Hồ Chí Minh",
    "Hồ Chí Minh.": "Hồ Chí Minh",
    "TP. Cam Ranh": "Khánh Hòa", "Quận Nam Từ Liêm": "Hà Nội"
}
df["Province"] = df["Province"].replace(province_mapping)

# ================== Remove invalid provinces ==================
valid_provinces = [
    "Hà Nội","Hồ Chí Minh","Khánh Hòa","Bình Dương","Đà Nẵng","Khác"
]
df = df[df["Province"].isin(valid_provinces)]

# ================== Export locations.json ==================
locations = {}
for prov in sorted(df["Province"].unique().tolist()):
    cities = sorted(df.loc[df["Province"] == prov, "City"].dropna().unique().tolist())
    if not cities:
        cities = ["N/A"]
    else:
        bad_keywords = ["bán nhà", "giá", "phòng công chứng", "đường số"]
        cities = [c for c in cities if not any(bad.lower() in c.lower() for bad in bad_keywords)]
        if not cities:
            cities = ["N/A"]
        if "N/A" not in cities:
            cities.insert(0, "N/A")
    locations[prov] = cities

with open(LOCATIONS_FILE, "w", encoding="utf-8") as f:
    json.dump(locations, f, ensure_ascii=False, indent=4)

# ================== Export choices.json ==================
def add_na(values):
    values = [v for v in values if str(v).strip() not in ["", "nan", "None"]]
    if not values:
        return ["N/A"]
    values = sorted(values)
    if "N/A" not in values:
        values.insert(0, "N/A")
    return values

choices = {
    "house_directions": add_na(df['House direction'].dropna().unique().tolist()),
    "balcony_directions": add_na(df['Balcony direction'].dropna().unique().tolist()),
    "legal_statuses": add_na(df['Legal status'].dropna().unique().tolist()),
    "furniture_states": add_na(df['Furniture state'].dropna().unique().tolist())
}

with open(CHOICES_FILE, "w", encoding="utf-8") as f:
    json.dump(choices, f, ensure_ascii=False, indent=4)

# ================== Handle missing ==================
fill_with_median = lambda x: x.fillna(x.median())
fill_with_mode = lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "N/A")

num_features = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
cat_features = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state', 'Province', 'City']

for col in num_features:
    df[col] = fill_with_median(df[col])
for col in cat_features:
    df[col] = fill_with_mode(df[col])

# ================== Features & Target ==================
X = df[num_features + cat_features]
y = df['Price']

# ================== Preprocessor ==================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# ================== Train/Test Split ==================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================== Models ==================
models = {
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}
try:
    from xgboost import XGBRegressor
    models["xgboost"] = XGBRegressor(random_state=42, n_jobs=-1)
except ImportError:
    print("⚠️ XGBoost chưa cài, bỏ qua.")

# ================== Hyperparameter grids ==================
param_grids = {
    "random_forest": {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5, 10],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
    },
}

# ================== Train + Evaluate ==================
results = []
os.makedirs("models", exist_ok=True)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    cv_mean, cv_std = scores.mean(), scores.std()
    print(f"📊 {name} CV R² = {cv_mean:.3f} ± {cv_std:.3f}")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"📌 {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

    joblib.dump(model, f"models/{name}.joblib")

    return {
        "model": name,
        "CV_R2_mean": round(cv_mean, 3),
        "CV_R2_std": round(cv_std, 3),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3)
    }

for name, model in models.items():
    print(f"\n🔄 Training model: {name}")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    
    if name in param_grids:
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grids[name],
            n_iter=10,
            scoring="r2",
            cv=cv,
            n_jobs=1,
            random_state=42
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"🔍 {name} best params: {search.best_params_}")
    else:
        best_model = pipe.fit(X_train, y_train)
    
    metrics = evaluate_model(name, best_model, X_train, y_train, X_test, y_test)
    results.append(metrics)

results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
print("\n✅ Training xong, models đã lưu trong thư mục models/ và kết quả trong results.csv")
