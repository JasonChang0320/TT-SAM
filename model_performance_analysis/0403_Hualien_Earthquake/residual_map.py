import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
from obspy.taup.taup_geo import calc_dist
from obspy.geodetics import degrees2kilometers

sys.path.append("..")
from analysis import Consider_Angle


def calculate_angle(x1, y1, x2, y2):
    # 計算兩點之間的斜率
    delta_x = x2 - x1
    delta_y = y2 - y1
    slope = delta_y / delta_x

    # 使用反正切函數計算角度（以弧度為單位）
    angle_radians = math.atan(slope)

    # 將弧度轉換為角度
    angle_degrees = math.degrees(angle_radians)

    # 將角度調整為0到360度的範圍
    if delta_x < 0:
        angle_degrees += 180
    elif delta_x >= 0 and delta_y < 0:
        angle_degrees += 360

    return angle_degrees % 360


answer = pd.read_csv(f"true_answer.csv")
prediction_3 = pd.read_csv(f"no_include_broken_data_prediction/3_sec_prediction.csv")
prediction_5 = pd.read_csv(f"no_include_broken_data_prediction/5_sec_prediction.csv")
prediction_7 = pd.read_csv(f"no_include_broken_data_prediction/7_sec_prediction.csv")
prediction_10 = pd.read_csv(f"no_include_broken_data_prediction/10_sec_prediction.csv")

max_prediction = pd.concat(
    [
        prediction_3,
        prediction_5["predict"],
        prediction_7["predict"],
        prediction_10["predict"],
    ],
    axis=1,
)

max_prediction.columns = [
    "3_predict",
    "station_name",
    "latitude",
    "longitude",
    "elevation",
    "5_predict",
    "7_predict",
    "10_predict",
]
max_prediction["max_predict"] = max_prediction.apply(
    lambda row: max(
        row["3_predict"], row["5_predict"], row["7_predict"], row["10_predict"]
    ),
    axis=1,
)

max_prediction = pd.merge(
    answer, max_prediction, how="left", left_on="location_code", right_on="station_name"
)
max_prediction.dropna(inplace=True)

init_latitude = max_prediction.query("location_code=='HWA074'")["latitude"].values[0]
init_longitude = max_prediction.query("location_code=='HWA074'")["longitude"].values[0]
event_lat = 23.77
event_lon = 121.67

max_prediction = max_prediction.reset_index(drop=True)
flattening_of_planet = 1 / 298.257223563

for i in range(len(max_prediction)):
    lat = max_prediction["latitude"][i]
    lon = max_prediction["longitude"][i]
    angle = calculate_angle(init_longitude, init_latitude, lon, lat)
    epi_dist = degrees2kilometers(
        calc_dist(
            event_lat,
            event_lon,
            lat,
            lon,
            radius_of_planet_in_km=6371.0,
            flattening_of_planet=flattening_of_planet,
        )
    )
    max_prediction.loc[i, "angle"] = angle
    max_prediction.loc[i, "dist"] = epi_dist

fig, ax = Consider_Angle.plot_pga_attenuation(prediction=max_prediction)
# fig.savefig("PGA_attenuation.png",dpi=300)

fig, ax = Consider_Angle.angle_map(
    stations=max_prediction,
    init_sta_lat=init_latitude,
    init_sta_lon=init_longitude,
    event_lat=23.77,
    event_lon=121.66,
)
# fig.savefig("Angle_map.png",dpi=300)