import pandas as pd
import numpy as np
from numpy import sin, cos, tan, radians
import math
from tqdm import tqdm


def lonlat_to_97(lon, lat):
    """
    It transforms longitude, latitude to TWD97 system.

    Parameters
    ----------
    lon : float
        longitude in degrees
    lat : float
        latitude in degrees

    Returns
    -------
    x, y [TWD97]
    """

    lat = radians(lat)
    lon = radians(lon)

    a = 6378137.0
    b = 6356752.314245
    long0 = radians(121)
    k0 = 0.9999
    dx = 250000

    e = (1 - b**2 / a**2) ** 0.5
    e2 = e**2 / (1 - e**2)
    n = (a - b) / (a + b)
    nu = a / (1 - (e**2) * (sin(lat) ** 2)) ** 0.5
    p = lon - long0

    A = a * (1 - n + (5 / 4.0) * (n**2 - n**3) + (81 / 64.0) * (n**4 - n**5))
    B = (3 * a * n / 2.0) * (
        1 - n + (7 / 8.0) * (n**2 - n**3) + (55 / 64.0) * (n**4 - n**5)
    )
    C = (15 * a * (n**2) / 16.0) * (1 - n + (3 / 4.0) * (n**2 - n**3))
    D = (35 * a * (n**3) / 48.0) * (1 - n + (11 / 16.0) * (n**2 - n**3))
    E = (315 * a * (n**4) / 51.0) * (1 - n)

    S = (
        A * lat
        - B * sin(2 * lat)
        + C * sin(4 * lat)
        - D * sin(6 * lat)
        + E * sin(8 * lat)
    )

    K1 = S * k0
    K2 = k0 * nu * sin(2 * lat) / 4.0
    K3 = (k0 * nu * sin(lat) * (cos(lat) ** 3) / 24.0) * (
        5 - tan(lat) ** 2 + 9 * e2 * (cos(lat) ** 2) + 4 * (e2**2) * (cos(lat) ** 4)
    )

    y_97 = K1 + K2 * (p**2) + K3 * (p**4)

    K4 = k0 * nu * cos(lat)
    K5 = (k0 * nu * (cos(lat) ** 3) / 6.0) * (1 - tan(lat) ** 2 + e2 * (cos(lat) ** 2))

    x_97 = K4 * p + K5 * (p**3) + dx
    return x_97, y_97


def twd67_to_97(x, y):
    """_summary_


    Parameters
    ----------
    x : float
        x in TWD67 system
    y : float
        x in TWD67 system

    Returns
    -------
    x and y in TWD97 system
    """
    A = 0.00001549
    B = 0.000006521

    x_97 = x + 807.8 + A * x + B * y
    y_97 = y - 248.6 + A * y + B * x
    return x_97, y_97


def twd97_to_lonlat(x=174458.0, y=2525824.0):
    """
    Parameters
    ----------
    x : float
        TWD97 coord system. The default is 174458.0.
    y : float
        TWD97 coord system. The default is 2525824.0.
    Returns
    -------
    list
        [longitude, latitude]
    """

    a = 6378137
    b = 6356752.314245
    long_0 = 121 * math.pi / 180.0
    k0 = 0.9999
    dx = 250000
    dy = 0

    e = math.pow((1 - math.pow(b, 2) / math.pow(a, 2)), 0.5)

    x -= dx
    y -= dy

    M = y / k0

    mu = M / (
        a
        * (1 - math.pow(e, 2) / 4 - 3 * math.pow(e, 4) / 64 - 5 * math.pow(e, 6) / 256)
    )
    e1 = (1.0 - pow((1 - pow(e, 2)), 0.5)) / (
        1.0 + math.pow((1.0 - math.pow(e, 2)), 0.5)
    )

    j1 = 3 * e1 / 2 - 27 * math.pow(e1, 3) / 32
    j2 = 21 * math.pow(e1, 2) / 16 - 55 * math.pow(e1, 4) / 32
    j3 = 151 * math.pow(e1, 3) / 96
    j4 = 1097 * math.pow(e1, 4) / 512

    fp = (
        mu
        + j1 * math.sin(2 * mu)
        + j2 * math.sin(4 * mu)
        + j3 * math.sin(6 * mu)
        + j4 * math.sin(8 * mu)
    )

    e2 = math.pow((e * a / b), 2)
    c1 = math.pow(e2 * math.cos(fp), 2)
    t1 = math.pow(math.tan(fp), 2)
    r1 = (
        a
        * (1 - math.pow(e, 2))
        / math.pow((1 - math.pow(e, 2) * math.pow(math.sin(fp), 2)), (3 / 2))
    )
    n1 = a / math.pow((1 - math.pow(e, 2) * math.pow(math.sin(fp), 2)), 0.5)
    d = x / (n1 * k0)

    q1 = n1 * math.tan(fp) / r1
    q2 = math.pow(d, 2) / 2
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * math.pow(c1, 2) - 9 * e2) * math.pow(d, 4) / 24
    q4 = (
        (
            61
            + 90 * t1
            + 298 * c1
            + 45 * math.pow(t1, 2)
            - 3 * math.pow(c1, 2)
            - 252 * e2
        )
        * math.pow(d, 6)
        / 720
    )
    lat = fp - q1 * (q2 - q3 + q4)

    q5 = d
    q6 = (1 + 2 * t1 + c1) * math.pow(d, 3) / 6
    q7 = (
        (5 - 2 * c1 + 28 * t1 - 3 * math.pow(c1, 2) + 8 * e2 + 24 * math.pow(t1, 2))
        * math.pow(d, 5)
        / 120
    )
    lon = long_0 + (q5 - q6 + q7) / math.cos(fp)

    lat = (lat * 180) / math.pi
    lon = (lon * 180) / math.pi
    return [lon, lat]


def find_nearest_point(target_point, points):
    """
    找到點集points中距離目標點target_point最近的點。

    參數：
    target_point: 目標點的坐標，一個包含兩個元素的列表或元組，例如 [x, y]。
    points: 點集，一個包含多個點坐標的二維數組，每個點為一個包含兩個元素的列表或元組，例如 [[x1, y1], [x2, y2], ...]。

    返回值：
    nearest_point: 距離目標點最近的點的坐標，一個包含兩個元素的列表，例如 [x_nearest, y_nearest]。
    """
    target_point = np.array(target_point)
    points = np.array(points)

    # 使用cdist計算距離矩陣
    distances = distance.cdist([target_point], points)

    # 找到距離最小的點的索引
    nearest_index = np.argmin(distances)

    # 返回距離最小的點的坐標
    nearest_point = points[nearest_index]

    return nearest_index, nearest_point


def get_unique_with_other_columns(group):
    # 獲取唯一值
    unique_value = group["station_name"].unique()[0]
    # 獲取其他columns的值
    other_columns_values = group.drop_duplicates(subset=["station_name"])
    return other_columns_values


sta_path = "../data/station_information"
start_year = 1999
end_year = 2019
trace = pd.read_csv(f"./events_traces_catalog/{start_year}_{end_year}_final_traces.csv")
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
vs30_info = pd.read_csv(f"{sta_path}/egdt_TSMIP_station_vs30.csv")

merge_traces = pd.merge(
    trace,
    station_info[["location_code", "latitude", "longitude", "elevation (m)"]],
    how="left",
    left_on="station_name",
    right_on="location_code",
)

merge_traces = pd.merge(
    merge_traces,
    vs30_info[["station_code", "Vs30"]],
    how="left",
    left_on="station_name",
    right_on="station_code",
)

noVs30_station_value_counts = (
    merge_traces[merge_traces["Vs30"].isna()]["station_name"]
    .value_counts()
    .rename_axis("station_name")
    .reset_index(name="counts")
)
noVs30_station_value_counts = pd.merge(
    noVs30_station_value_counts,
    station_info[["location_code", "latitude", "longitude", "elevation (m)"]],
    how="left",
    left_on="station_name",
    right_on="location_code",
)
Vs30_station_value_counts = (
    merge_traces[~merge_traces["Vs30"].isna()]["station_name"]
    .value_counts()
    .rename_axis("station_name")
    .reset_index(name="counts")
)
Vs30_station_value_counts = pd.merge(
    Vs30_station_value_counts,
    station_info[["location_code", "latitude", "longitude", "elevation (m)"]],
    how="left",
    left_on="station_name",
    right_on="location_code",
)


import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# plot station map with vs30 or not
src_crs = ccrs.PlateCarree()
fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
ax_map.coastlines("10m")
ax_map.scatter(
    Vs30_station_value_counts["longitude"],
    Vs30_station_value_counts["latitude"],
    edgecolors="k",
    linewidth=1,
    marker="o",
    s=10,
    zorder=3,
    label="include Vs30",
    alpha=0.5,
)
ax_map.scatter(
    noVs30_station_value_counts["longitude"],
    noVs30_station_value_counts["latitude"],
    edgecolors="k",
    linewidth=1,
    marker="o",
    s=10,
    zorder=3,
    label="No Vs30",
)
ax_map.set_title("Vs30 from egdt")
ax_map.legend()
# fig.savefig("./events_traces_catalog/Vs30 map.png",dpi=300)

import pygmt
import os


def grd_to_xyz(input_grd, output_xyz):
    with pygmt.clib.Session() as session:
        # 使用pygmt.grd2xyz進行轉換
        session.call_module("grd2xyz", f"{input_grd} > {output_xyz}")


file_path = "../data/station_information"

# transfer grd file to xyz file
# if __name__ == "__main__":
#     os.getcwd()
#     input_grd_file = f"{file_path}/Vs30ofTaiwan.grd"  # 輸入GRD檔案的路徑
#     output_xyz_file = f"{file_path}/Vs30ofTaiwan.xyz"  # 輸出XYZ檔案的路徑

#     grd_to_xyz(input_grd_file, output_xyz_file)
xyz_file = f"{file_path}/Vs30ofTaiwan.xyz"
vs30_table = pd.read_csv(xyz_file, sep="\s+", header=None, names=["x", "y", "Vs30"])
vs30_table.dropna(inplace=True)
vs30_table.reset_index(drop=True, inplace=True)
plt.scatter(vs30_table["x"], vs30_table["y"])

# transform coordinate
vs30_table["x_97"], vs30_table["y_97"] = twd67_to_97(vs30_table["x"], vs30_table["y"])
vs30_table["lon"] = 0
vs30_table["lat"] = 0
for i in tqdm(range(len(vs30_table))):
    vs30_table["lon"][i], vs30_table["lat"][i] = twd97_to_lonlat(
        vs30_table["x_97"][i], vs30_table["y_97"][i]
    )
# vs30_table.to_csv(f"{file_path}/Vs30ofTaiwan.csv",index=False)

# vs30 map fill into no vs30 station
vs30_table = pd.read_csv(f"{file_path}/Vs30ofTaiwan.csv")
target_points = noVs30_station_value_counts[["longitude", "latitude"]].values.tolist()
points = vs30_table[["lon", "lat"]].values.tolist()
import numpy as np
from scipy.spatial import distance

referenced_table = {
    "index": [],
    "Vs30 referenced lon": [],
    "Vs30 referenced lat": [],
    "Vs30": [],
}

for target_point in tqdm(target_points):
    nearest_index, nearest_point = find_nearest_point(target_point, points)

    referenced_table["index"].append(nearest_index)
    referenced_table["Vs30 referenced lon"].append(nearest_point[0])
    referenced_table["Vs30 referenced lat"].append(nearest_point[1])
    referenced_table["Vs30"].append(vs30_table.loc[nearest_index]["Vs30"])

for key in referenced_table.keys():
    noVs30_station_value_counts[f"{key}"] = referenced_table[f"{key}"]

fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
ax_map.coastlines("10m")
ax_map.scatter(
    noVs30_station_value_counts["Vs30 referenced lon"],
    noVs30_station_value_counts["Vs30 referenced lat"],
    edgecolors="k",
    linewidth=1,
    marker="o",
    s=40,
    label="referenced",
    alpha=0.5,
)
ax_map.scatter(
    noVs30_station_value_counts["longitude"],
    noVs30_station_value_counts["latitude"],
    edgecolors="k",
    linewidth=1,
    marker="o",
    s=10,
    label="No Vs30",
)
ax_map.set_title("Vs30 filled from Lee's map")
ax_map.legend()
# fig.savefig("./events_traces_catalog/Vs30 filled from Lee map.png",dpi=300)

# fill vs30 into traces table
merge_traces[merge_traces["Vs30"].isna()]

for index in merge_traces[merge_traces["Vs30"].isna()].index:
    station_name = merge_traces.iloc[index]["station_name"]
    vs30 = np.round(
        noVs30_station_value_counts.query(f"station_name=='{station_name}'")[
            "Vs30"
        ].values[0],
        2,
    )
    merge_traces.loc[index, "Vs30"] = vs30
    print(
        station_name,
        noVs30_station_value_counts.query(f"station_name=='{station_name}'")[
            "Vs30"
        ].values,
    )

merge_traces.drop(["location_code", "station_code"], axis=1, inplace=True)
# merge_traces.to_csv(
#     f"./events_traces_catalog/{start_year}_{end_year}_final_traces_Vs30.csv",
#     index=False,
# )

# plot final vs30 value map
trace_with_vs30 = pd.read_csv(
    f"./events_traces_catalog/{start_year}_{end_year}_final_traces_Vs30.csv"
)
vs30_table = trace_with_vs30[["station_name", "longitude", "latitude", "Vs30"]]
vs30_table = (
    vs30_table.groupby("station_name")
    .apply(get_unique_with_other_columns)
    .reset_index(drop=True)
)

src_crs = ccrs.PlateCarree()
fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
ax_map.coastlines("10m")
map = ax_map.scatter(
    vs30_table["longitude"],
    vs30_table["latitude"],
    linewidth=1,
    marker="o",
    s=10,
    c=vs30_table["Vs30"],
    cmap="copper_r",
)
ax_map.set_title("Final Vs30 Map")
cbar = plt.colorbar(map, ax=ax_map)
cbar.set_label("Vs30 (m/s)")
# fig.savefig("./events_traces_catalog/Final Vs30 Map.png",dpi=300)
