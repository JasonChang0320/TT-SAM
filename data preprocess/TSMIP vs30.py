import pandas as pd
import numpy as np

sta_path = "../data/station_information"
start_year = 1999
end_year = 2019
trace = pd.read_csv(f"./events_traces_catalog/{start_year}_{end_year}_final_traces.csv")
# trace=trace[trace["pga"]>=np.log10(0.25)]
# trace=trace[trace["intensity"]>=4]
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
noVs30_station_value_counts=pd.merge(noVs30_station_value_counts,station_info[["location_code", "latitude", "longitude", "elevation (m)"]],
    how="left",
    left_on="station_name",
    right_on="location_code")

total_station_value_counts=(
    merge_traces["station_name"]
    .value_counts()
    .rename_axis("station_name")
    .reset_index(name="counts")
)
total_station_value_counts=pd.merge(total_station_value_counts,station_info[["location_code", "latitude", "longitude", "elevation (m)"]],
    how="left",
    left_on="station_name",
    right_on="location_code")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# plot station map with vs30 or not & check number of records received by stations
src_crs = ccrs.PlateCarree()
fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
ax_map.coastlines("10m")
ax_map.scatter(
    total_station_value_counts["longitude"],
    total_station_value_counts["latitude"],
    edgecolors="k",
    linewidth=1,
    marker="o",
    s=total_station_value_counts["counts"]/2,
    # s=10,
    zorder=3,
    # label="include Vs30",
    alpha=0.5
)
# ax_map.scatter(
#     noVs30_station_value_counts["longitude"],
#     noVs30_station_value_counts["latitude"],
#     # c=true_label,
#     # cmap=cmap,
#     # norm=norm,
#     edgecolors="k",
#     linewidth=1,
#     marker="o",
#     s=10,
#     zorder=3,
#     label="No Vs30",
# )
ax_map.set_title("Number of records received by stations")
# ax_map.legend()
# fig.savefig("./events_traces_catalog/Number of records received by stations.png",dpi=300)

#fill Vs30 missing value
import pygmt
import os

def grd_to_xyz(input_grd, output_xyz):
    with pygmt.clib.Session() as session:
        # 使用pygmt.grd2xyz進行轉換
        session.call_module("grd2xyz", f"{input_grd} > {output_xyz}")

file_path = '../data/station_information'

if __name__ == "__main__":
    os.getcwd()
    input_grd_file = f"{file_path}/Vs30ofTaiwan.grd"  # 輸入GRD檔案的路徑
    output_xyz_file = f"{file_path}/Vs30ofTaiwan.xyz"  # 輸出XYZ檔案的路徑

    grd_to_xyz(input_grd_file, output_xyz_file)

vs30_table=pd.read_csv(output_xyz_file,sep="\s+",header=None,names=["x","y","Vs30"])
vs30_table.dropna(inplace=True)

plt.plot(vs30_table["x"],vs30_table["y"])



