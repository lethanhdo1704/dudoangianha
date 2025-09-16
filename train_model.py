import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib
import json

# ================== Load data ==================
file_path = "data/vietnam_housing_dataset.csv"
df = pd.read_csv(file_path)

# ================== Tách tỉnh từ cột Address ==================
df["Province"] = df["Address"].apply(lambda x: x.split(",")[-1].strip() if pd.notnull(x) else "Khác")

# ================== Chuẩn hóa tên tỉnh/thành ==================
province_mapping = {
    "HN": "Hà Nội", "Hà Nội": "Hà Nội", "Hà Nội.": "Hà Nội",
    "TP Hồ Chí Minh": "Hồ Chí Minh", "TP. HCM": "Hồ Chí Minh",
    "TPHCM": "Hồ Chí Minh", "TpHCM": "Hồ Chí Minh",
    "Hồ Chí Minh.": "Hồ Chí Minh", "Hồ Chí Mính": "Hồ Chí Minh",
    "Hồ Chí Minh giá 2tỷ380": "Hồ Chí Minh", "Quận 8": "Hồ Chí Minh",
    "Quận Bình Thạnh": "Hồ Chí Minh", "TP. Cam Ranh": "Khánh Hòa",
    "Quận Nam Từ Liêm": "Hà Nội",
    "Bà Rịa Vũng Tàu.": "Bà Rịa Vũng Tàu", "Bình Dương.": "Bình Dương",
    "Bình Dương (gần cafe Xóm Vắng 2)": "Bình Dương", "Bình Phước.": "Bình Phước",
    "Bình Thuận.": "Bình Thuận", "Bình Định.": "Bình Định",
    "Bạc Liêu.": "Bạc Liêu", "Bắc Giang.": "Bắc Giang",
    "Bắc Ninh.": "Bắc Ninh", "Bến Tre.": "Bến Tre", "Cần Thơ.": "Cần Thơ",
    "Hưng Yên.": "Hưng Yên", "Hải Phòng.": "Hải Phòng",
    "Khánh Hòa.": "Khánh Hòa", "Kiên Giang.": "Kiên Giang",
    "Kon Tum.": "Kon Tum", "Long An.": "Long An", "Lào Cai.": "Lào Cai",
    "Lâm Đồng.": "Lâm Đồng", "Phú Thọ.": "Phú Thọ", "Phú Yên.": "Phú Yên",
    "Quảng Ninh.": "Quảng Ninh", "Quảng Ninh (Ngã 3 đường Hòn Gai cũ)": "Quảng Ninh",
    "Thanh Hóa.": "Thanh Hóa", "Thái Nguyên.": "Thái Nguyên",
    "Thừa Thiên Huế.": "Thừa Thiên Huế", "Trà Vinh.": "Trà Vinh",
    "Đà Nẵng.": "Đà Nẵng", "Đắk Lắk.": "Đắk Lắk", "Đồng Nai.": "Đồng Nai",
    "giá 6ty": "Khác", "": "Khác"
}
df["Province"] = df["Province"].replace(province_mapping)

valid_provinces = [
    "An Giang","Bà Rịa Vũng Tàu","Bình Dương","Bình Phước","Bình Thuận","Bình Định",
    "Bạc Liêu","Bắc Giang","Bắc Ninh","Bến Tre","Cà Mau","Cần Thơ","Gia Lai",
    "Hà Giang","Hà Nam","Hà Nội","Hà Tĩnh","Hòa Bình","Hưng Yên","Hải Dương",
    "Hải Phòng","Hậu Giang","Hồ Chí Minh","Khánh Hòa","Kiên Giang","Kon Tum",
    "Long An","Lào Cai","Lâm Đồng","Lạng Sơn","Nam Định","Nghệ An","Ninh Bình",
    "Ninh Thuận","Phú Thọ","Phú Yên","Quảng Bình","Quảng Nam","Quảng Ngãi",
    "Quảng Ninh","Quảng Trị","Sóc Trăng","Sơn La","Thanh Hóa","Thái Bình",
    "Thái Nguyên","Thừa Thiên Huế","Tiền Giang","Trà Vinh","Tuyên Quang",
    "Tây Ninh","Vĩnh Long","Vĩnh Phúc","Yên Bái","Điện Biên","Đà Nẵng",
    "Đắk Lắk","Đồng Nai","Đồng Tháp","Khác"
]
df["Province"] = df["Province"].apply(lambda x: x if x in valid_provinces else "Khác")

# Xuất danh sách tỉnh đã chuẩn hóa
provinces = sorted(df["Province"].dropna().unique().tolist())
with open("provinces.json", "w", encoding="utf-8") as f:
    json.dump(provinces, f, ensure_ascii=False, indent=4)
print("✅ Đã chuẩn hóa và lưu danh sách tỉnh vào provinces.json")

# ================== Xử lý dữ liệu thiếu ==================
fill_with_mode = lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
fill_with_median = lambda x: x.fillna(x.median())

df['Frontage'] = df.groupby('Address')['Frontage'].transform(fill_with_mode)
df['Access Road'] = df.groupby('Address')['Access Road'].transform(fill_with_mode)
df['Floors'] = df.groupby('Address')['Floors'].transform(fill_with_mode)

df['Floors'].fillna(df['Floors'].median(), inplace=True)
df['House direction'].fillna('N/A', inplace=True)
df['Balcony direction'].fillna('N/A', inplace=True)

df['Bedrooms'] = df.groupby('Address')['Bedrooms'].transform(fill_with_median)
df['Bathrooms'] = df.groupby('Address')['Bathrooms'].transform(fill_with_median)
df['Bedrooms'].fillna(df['Bedrooms'].median(), inplace=True)
df['Bathrooms'].fillna(df['Bathrooms'].median(), inplace=True)

df['Legal status'].fillna('Sale contract', inplace=True)
df['Furniture state'].fillna('N/A', inplace=True)

# ================== Xuất danh sách lựa chọn cho dropdown ==================
house_directions = sorted(df['House direction'].dropna().unique().tolist())
balcony_directions = sorted(df['Balcony direction'].dropna().unique().tolist())
legal_statuses = sorted(df['Legal status'].dropna().unique().tolist())
furniture_states = sorted(df['Furniture state'].dropna().unique().tolist())

choices = {
    "house_directions": house_directions,
    "balcony_directions": balcony_directions,
    "legal_statuses": legal_statuses,
    "furniture_states": furniture_states
}
with open("choices.json", "w", encoding="utf-8") as f:
    json.dump(choices, f, ensure_ascii=False, indent=4)
print("✅ Đã lưu choices.json cho dropdown")

# ================== Features & Target ==================
num_features = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
cat_features = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state', 'Province']

X = df[num_features + cat_features]
y = df['Price']   # Giá đã là TỶ trong dataset gốc

# ================== Preprocessor ==================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
])

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
    """Train, evaluate and save model + metrics"""
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    cv_mean, cv_std = scores.mean(), scores.std()
    print(f"📊 {name} CV R² = {cv_mean:.3f} ± {cv_std:.3f}")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"📌 {name}: MAE={mae:.2f} tỷ, RMSE={rmse:.2f} tỷ, R²={r2:.3f}")

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
