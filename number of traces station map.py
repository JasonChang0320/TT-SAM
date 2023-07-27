import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

sta_path = "data/station_information"
prediction = pd.read_csv(
    "predict/acc predict pga 1999_2019/model 2 5 sec prediction (train dataset).csv"
)
prediction = prediction[prediction["predict"] >= np.log10(0.25)]
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
# plot number of traces received by stations in 2016
merge_traces = pd.merge(
    prediction,
    station_info[["location_code", "latitude", "longitude", "elevation (m)"]],
    how="left",
    left_on=["latitude", "longitude", "elevation"],
    right_on=["latitude", "longitude", "elevation (m)"],
)
total_station_value_counts = (
    merge_traces["location_code"]
    .value_counts()
    .rename_axis("location_code")
    .reset_index(name="counts")
)
total_station_value_counts = pd.merge(
    total_station_value_counts,
    station_info[["location_code", "latitude", "longitude", "elevation (m)"]],
    how="left",
    left_on="location_code",
    right_on="location_code",
)
src_crs = ccrs.PlateCarree()
fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
ax_map.coastlines("10m")
ax_map.scatter(
    total_station_value_counts["longitude"],
    total_station_value_counts["latitude"],
    edgecolors="k",
    linewidth=1,
    marker="o",
    s=total_station_value_counts["counts"] / 2,
    zorder=3,
    alpha=0.5,
)
ax_map.set_title(
    "Number of records (predict bigger than 25 gal) received by stations in train data"
)
# fig.savefig("./data preprocess/events_traces_catalog/Number of records (predict bigger than 25 gal) received by stations in train data.png")
