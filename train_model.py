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
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# ================== Train/Test Split ==================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================== Models ==================
models = {
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(alpha=1.0),
    # Random Forest full performance
    "random_forest": RandomForestRegressor(
        n_estimators=150,   # giảm từ 300
        max_depth=15,       # giới hạn độ sâu để giảm memory
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ),

    # Gradient Boosting full performance
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        random_state=42
    ),
}

# Try XGBoost for max accuracy
try:
    from xgboost import XGBRegressor
    models["xgboost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        n_jobs=-1
    )
except ImportError:
    print("⚠️ XGBoost chưa cài, bỏ qua.")

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

    # Compress khi lưu để giảm dung lượng
    joblib.dump(model, f"models/{name}.joblib", compress=3)

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
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    best_model = pipe.fit(X_train, y_train)
    
    metrics = evaluate_model(name, best_model, X_train, y_train, X_test, y_test)
    results.append(metrics)

# ================== Save Results ==================
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
print("\n✅ Training xong, models đã lưu trong thư mục models/ và kết quả trong results.csv")
