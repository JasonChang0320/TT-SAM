import pandas as pd
from visualize import plot_received_traces_station_map

sta_path = "data/station_information"
input_path = "predict/station_blind_Vs30_bias2closed_station_2016"
output_path = "./data preprocess/events_traces_catalog"
prediction = pd.read_csv(f"{input_path}/model 11 5 sec prediction.csv")
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
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
title = "Number of records received by stations in test data"
fig, ax = plot_received_traces_station_map(total_station_value_counts, title=title)

# total_station_value_counts.to_csv(
#     "predict/station_blind_Vs30_bias2closed_station_2016/Number of records received by stations in train data.csv",
#     index=False,
# )
