import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


def plot_received_traces_station_map(
    total_station_value_counts, title="Received traces map", output_path=None
):
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
    ax_map.set_title(f"{title}")
    if output_path:
        fig.savefig(f"{output_path}/{title}.png", dpi=300)
    return fig, ax_map


sta_path = "data/station_information"
input_path = "predict/station_blind_Vs30_bias2closed_station_2016"
output_path = "./data preprocess/events_traces_catalog"
prediction = pd.read_csv(f"{input_path}/model 11 5 sec prediction (train_data).csv")
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
# total_station_value_counts.to_csv(
#     "predict/station_blind_Vs30_bias2closed_station_2016/Number of records received by stations in train data.csv",
#     index=False,
# )
fig_title = (
    "Number of records (predict bigger than 25 gal) received by stations in train data"
)
fig, ax = plot_received_traces_station_map(total_station_value_counts, title=fig_title)
