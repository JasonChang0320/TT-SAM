import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

test_year = 2016
mask_sec = 3
Afile_path = "data preprocess/events_traces_catalog"
prediction = pd.read_csv(
    f"predict/station_blind_noVs30_bias2closed_station_{test_year}/{mask_sec} sec ensemble 510.csv"
)

catalog = pd.read_csv(f"{Afile_path}/1999_2019_final_catalog.csv")
traces_info = pd.read_csv(f"{Afile_path}/1999_2019_final_traces_Vs30.csv")

trace_merge_catalog = pd.merge(
    traces_info,
    catalog[
        [
            "EQ_ID",
            "lat",
            "lat_minute",
            "lon",
            "lon_minute",
            "depth",
            "magnitude",
            "nsta",
            "nearest_sta_dist (km)",
        ]
    ],
    on="EQ_ID",
    how="left",
)
trace_merge_catalog["event_lat"] = (
    trace_merge_catalog["lat"] + trace_merge_catalog["lat_minute"] / 60
)

trace_merge_catalog["event_lon"] = (
    trace_merge_catalog["lon"] + trace_merge_catalog["lon_minute"] / 60
)
trace_merge_catalog.drop(
    ["lat", "lat_minute", "lon", "lon_minute"], axis=1, inplace=True
)
trace_merge_catalog.rename(columns={"elevation (m)": "elevation"}, inplace=True)

import h5py

# for EQ_ID in prediction["EQ_ID"].unique():

data_path = "D:/TEAM_TSMIP/data/TSMIP_1999_2019.hdf5"
dataset = h5py.File(data_path, "r")
for eq_id in prediction["EQ_ID"].unique():
    eq_id = int(eq_id)
    station_name = dataset["data"][str(eq_id)]["station_name"][:].tolist()

    prediction.loc[
        prediction.query(f"EQ_ID=={eq_id}").index, "station_name"
    ] = station_name

prediction["station_name"] = prediction["station_name"].str.decode("utf-8")


prediction_with_info = pd.merge(
    prediction,
    trace_merge_catalog.drop(
        [
            "latitude",
            "longitude",
            "elevation",
        ],
        axis=1,
    ),
    on=["EQ_ID", "station_name"],
    how="left",
    suffixes=["_window", "_file"],
)
# prediction_with_info.to_csv(f"predict/station_blind_noVs30_bias2closed_station_{test_year}/{mask_sec} sec ensemble with all info.csv",index=False)
miss_alarm = (prediction_with_info["predict"] < np.log10(0.25)) & (
    prediction_with_info["answer"] >= np.log10(0.25)
)
false_alarm = (prediction_with_info["predict"] >= np.log10(0.25)) & (
    prediction_with_info["answer"] < np.log10(0.25)
)

wrong_predict = prediction_with_info[miss_alarm | false_alarm]

meinong_earthquake = prediction_with_info.query("EQ_ID==24784.0")

no_meinong_predict = prediction_with_info.query("EQ_ID!=24784.0")
for column in prediction_with_info.columns:
    fig, ax = plt.subplots()
    ax.scatter(
        wrong_predict[f"{column}"],
        wrong_predict["predict"] - wrong_predict["answer"],
        s=10,
        alpha=0.3,
        label="others",
    )
    ax.scatter(
        wrong_predict.query("EQ_ID==24784.0")[f"{column}"],
        wrong_predict.query("EQ_ID==24784.0")["predict"]
        - wrong_predict.query("EQ_ID==24784.0")["answer"],
        s=10,
        alpha=0.3,
        c="r",
        label="meinong eq",
    )
    residual_mean = np.round(
        (wrong_predict["predict"] - wrong_predict["answer"]).mean(), 3
    )
    residual_std = np.round(
        (wrong_predict["predict"] - wrong_predict["answer"]).std(), 3
    )
    wrong_predict_rate = np.round(len(wrong_predict) / len(prediction_with_info),3)
    wrong_predict_count =len(wrong_predict)
    ax.legend()
    ax.set_xlabel(f"{column}")
    ax.set_ylabel("predict-answer")
    ax.set_title(
        f"Predicted residual in {test_year} \n mean: {residual_mean}, std: {residual_std}, wrong rate: {wrong_predict_rate}"
    )

    # fig.savefig(
    #     f"predict/station_blind_noVs30_bias2closed_station_{test_year}/{mask_sec} sec residual plots/{column}.png",
    #     dpi=300,
    # )
