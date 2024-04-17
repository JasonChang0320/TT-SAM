import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl import ticker
import matplotlib.pyplot as plt
from Vs30_preprocess import *

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


file_path = "../data/station_information"
# transfer grd file to xyz file (run only one time)
# if __name__ == "__main__":
#     os.getcwd()
#     input_grd_file = f"{file_path}/Vs30ofTaiwan.grd"  # 輸入GRD檔案的路徑
#     output_xyz_file = f"{file_path}/Vs30ofTaiwan.xyz"  # 輸出XYZ檔案的路徑

#     grd_to_xyz(input_grd_file, output_xyz_file)
xyz_file = f"{file_path}/Vs30ofTaiwan.xyz"
vs30_table = pd.read_csv(xyz_file, sep="\s+", header=None, names=["x", "y", "Vs30"])
vs30_table.dropna(inplace=True)
vs30_table.reset_index(drop=True, inplace=True)

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
ax_map.add_feature(cartopy.feature.OCEAN, zorder=2, edgecolor="k")
# ax_map.set_title("Final Vs30 Map")
cbar = plt.colorbar(map, ax=ax_map)
cbar.set_label("Vs30 (m/s)")
xmin, xmax = ax_map.get_xlim()
ymin, ymax = ax_map.get_ylim()
xticks = ticker.LongitudeLocator(nbins=5)._raw_ticks(xmin, xmax)
yticks = ticker.LatitudeLocator(nbins=5)._raw_ticks(ymin, ymax)

ax_map.set_xticks(xticks, crs=ccrs.PlateCarree())
ax_map.set_yticks(yticks, crs=ccrs.PlateCarree())

ax_map.xaxis.set_major_formatter(ticker.LongitudeFormatter(zero_direction_label=True))
ax_map.yaxis.set_major_formatter(ticker.LatitudeFormatter())

ax_map.xaxis.set_ticks_position("both")
ax_map.yaxis.set_ticks_position("both")

ax_map.set_xlim(xmin, xmax)
ax_map.set_ylim(ymin, ymax)
# fig.savefig("./events_traces_catalog/Final Vs30 Map.png",dpi=300)
