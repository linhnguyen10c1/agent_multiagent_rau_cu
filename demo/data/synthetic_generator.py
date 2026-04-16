# tạo dữ liệu giả lập 
import numpy as np
import pandas as pd

np.random.seed(42)

N_DAYS = 90
BASE_DEMAND = {"ca_chua": 60, "xa_lach": 30, "dua_leo": 40}  # kg/ngày
BASE_COST   = {"ca_chua": 12000, "xa_lach": 18000, "dua_leo": 8000}  # đ/kg nhập
BASE_PRICE  = {"ca_chua": 25000, "xa_lach": 35000, "dua_leo": 18000}  # đ/kg bán

def generate_day(day_idx):
    weekday = day_idx % 7  # 0=Mon, 6=Sun
    is_weekend = weekday >= 5
    is_holiday = np.random.rand() < 0.05  # 5% là ngày lễ
    is_rainy = np.random.rand() < 0.3     # 30% ngày mưa
    temperature = np.random.uniform(22, 34)

    weekday_mult = 1.3 if is_weekend else 1.0
    weather_mult = 0.75 if is_rainy else 1.0
    holiday_mult = 1.5 if is_holiday else 1.0

    record = {
        "day": day_idx,
        "weekday": weekday,
        "is_weekend": int(is_weekend),
        "is_holiday": int(is_holiday),
        "is_rainy": int(is_rainy),
        "temperature": round(temperature, 1),
    }
    for veg, base in BASE_DEMAND.items():
        demand = base * weekday_mult * weather_mult * holiday_mult
        demand *= np.random.uniform(0.9, 1.1)  # noise
        record[f"demand_{veg}"] = round(demand, 1)
        record[f"cost_{veg}"]   = BASE_COST[veg]
        record[f"price_{veg}"]  = BASE_PRICE[veg]
    return record

df = pd.DataFrame([generate_day(i) for i in range(N_DAYS)])
df.to_csv("vegetable_90days.csv", index=False)
print(df.head())