import os
import json
import pandas as pd

# ------------------ Config ------------------
FILE_PATH = "data/vietnam_housing_dataset.csv"
LOCATIONS_FILE = "locations.json"
CHOICES_FILE = "choices.json"

# ------------------ Load data ------------------
df = pd.read_csv(FILE_PATH)

# ------------------ Extract City & Province ------------------
def split_address(addr: str):
    if pd.isnull(addr) or str(addr).strip() == "":
        return pd.Series({"City": "N/A", "Province": "Khác"})
    parts = [p.strip() for p in str(addr).split(",") if p.strip()]
    province = parts[-1] if len(parts) >= 1 else "Khác"
    city = parts[-2] if len(parts) >= 2 else "N/A"
    return pd.Series({"City": city, "Province": province})

df[["City", "Province"]] = df["Address"].apply(split_address)

# ------------------ Normalize Province ------------------
province_mapping = {
    "HN": "Hà Nội", "Hà Nội": "Hà Nội", "Hà Nội.": "Hà Nội",
    "TP Hồ Chí Minh": "Hồ Chí Minh", "TP. HCM": "Hồ Chí Minh",
    "TPHCM": "Hồ Chí Minh", "TpHCM": "Hồ Chí Minh",
    "Hồ Chí Minh.": "Hồ Chí Minh", "Hồ Chí Mính": "Hồ Chí Minh",
    "TP. Cam Ranh": "Khánh Hòa", "Quận Nam Từ Liêm": "Hà Nội",
    "Bà Rịa Vũng Tàu.": "Bà Rịa Vũng Tàu", "Bình Dương.": "Bình Dương",
    "Bình Phước.": "Bình Phước", "Bình Thuận.": "Bình Thuận",
    "Bình Định.": "Bình Định", "Bạc Liêu.": "Bạc Liêu",
    "Bắc Giang.": "Bắc Giang", "Bắc Ninh.": "Bắc Ninh",
    "Bến Tre.": "Bến Tre", "Cần Thơ.": "Cần Thơ", "Hưng Yên.": "Hưng Yên",
    "Hải Phòng.": "Hải Phòng", "Khánh Hòa.": "Khánh Hòa",
    "Kiên Giang.": "Kiên Giang", "Kon Tum.": "Kon Tum",
    "Long An.": "Long An", "Lào Cai.": "Lào Cai", "Lâm Đồng.": "Lâm Đồng",
    "Phú Thọ.": "Phú Thọ", "Phú Yên.": "Phú Yên", "Quảng Ninh.": "Quảng Ninh",
    "Thanh Hóa.": "Thanh Hóa", "Thái Nguyên.": "Thái Nguyên",
    "Thừa Thiên Huế.": "Thừa Thiên Huế", "Trà Vinh.": "Trà Vinh",
    "Đà Nẵng.": "Đà Nẵng", "Đắk Lắk.": "Đắk Lắk", "Đồng Nai.": "Đồng Nai",
}
df["Province"] = df["Province"].replace(province_mapping)

# ------------------ Remove invalid provinces ------------------
valid_provinces = set([
    "An Giang","Bà Rịa Vũng Tàu","Bắc Giang","Bắc Kạn","Bạc Liêu","Bắc Ninh",
    "Bến Tre","Bình Dương","Bình Định","Bình Phước","Bình Thuận","Cà Mau",
    "Cần Thơ","Cao Bằng","Đà Nẵng","Đắk Lắk","Đắk Nông","Điện Biên","Đồng Nai",
    "Đồng Tháp","Gia Lai","Hà Giang","Hà Nam","Hà Nội","Hà Tĩnh","Hải Dương",
    "Hải Phòng","Hậu Giang","Hòa Bình","Hưng Yên","Hồ Chí Minh","Khánh Hòa",
    "Kiên Giang","Kon Tum","Lai Châu","Lạng Sơn","Lào Cai","Lâm Đồng",
    "Long An","Nam Định","Nghệ An","Ninh Bình","Ninh Thuận","Phú Thọ",
    "Phú Yên","Quảng Bình","Quảng Nam","Quảng Ngãi","Quảng Ninh","Quảng Trị",
    "Sóc Trăng","Sơn La","Tây Ninh","Thái Bình","Thái Nguyên","Thanh Hóa",
    "Thừa Thiên Huế","Tiền Giang","Trà Vinh","Tuyên Quang","Vĩnh Long",
    "Vĩnh Phúc","Yên Bái"
])
df = df[df["Province"].isin(valid_provinces)]

# ------------------ Export locations.json ------------------
locations = {}
for prov in sorted(df["Province"].unique().tolist()):
    cities = sorted(df.loc[df["Province"] == prov, "City"].dropna().unique().tolist())
    if not cities:
        cities = ["N/A"]
    else:
        # bỏ city không hợp lệ
        bad_keywords = ["bán nhà", "giá", "phòng công chứng", "đường số"]
        cities = [c for c in cities if not any(bad.lower() in c.lower() for bad in bad_keywords)]
        if not cities:
            cities = ["N/A"]
        if "N/A" not in cities:
            cities.insert(0, "N/A")
    locations[prov] = cities

with open(LOCATIONS_FILE, "w", encoding="utf-8") as f:
    json.dump(locations, f, ensure_ascii=False, indent=4)
print(f"✅ Saved {LOCATIONS_FILE} (Province -> Cities)")

# ------------------ Export choices.json ------------------
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
print(f"✅ Saved {CHOICES_FILE}")
