import h5py
import numpy as np
import pandas as pd

from plot_predict_map import (
    correct_warning_with_epidist,
    plot_pga_map,
    true_predicted,
    warning_map,
    warning_time_hist,
)

mask_after_sec = 3
EQ_ID = 27305
trigger_station_threshold = 1
pga_threshold = np.log10(9.8 * 0.025)
path = "./predict/random sec updated dataset and new data generator/ok model prediction"
Afile_path = "data/Afile"

catalog = pd.read_csv(f"{Afile_path}/final catalog (station exist)_1.csv")
traces_info = pd.read_csv(
    f"{Afile_path}/1991-2020 traces with picking and label_new (sta location exist)_1.csv"
)
prediction = pd.read_csv(
    f"{path}/model2 7 9 {mask_after_sec} sec {trigger_station_threshold} triggered station prediction.csv"
)
station_info = pd.read_csv("data/station information/TSMIPstations_new.csv")

data_path = "D:/TEAM_TSMIP/data/TSMIP_new.hdf5"
dataset = h5py.File(data_path, "r")
p_picks = dataset["data"][str(EQ_ID)]["p_picks"][:].tolist()
pga_time = dataset["data"][str(EQ_ID)]["pga_time"][:].tolist()
station_name = dataset["data"][str(EQ_ID)]["station_name"][:].tolist()
latitude = dataset["data"][str(EQ_ID)]["station_location"][:, 0].tolist()
longitude = dataset["data"][str(EQ_ID)]["station_location"][:, 1].tolist()
elevation = dataset["data"][str(EQ_ID)]["station_location"][:, 2].tolist()
pga = dataset["data"][str(EQ_ID)]["pga"][:].tolist()

station_df = pd.DataFrame(
    {
        "p_picks": p_picks,
        "pga": pga,
        "pga_time": pga_time,
        "station_name": station_name,
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
    }
)
dataset.close()

station_df["station_name"] = station_df["station_name"].str.decode("utf-8")
station_df = station_df.drop_duplicates(
    subset=["p_picks", "pga_time", "station_name"], keep="last"
)

event_prediction = prediction[prediction["EQ_ID"] == EQ_ID]
event_prediction = event_prediction.drop_duplicates(
    subset=["p_picks", "pga_time", "predict"], keep="last"
)

event_prediction = pd.merge(
    event_prediction,
    station_df[["p_picks", "pga_time", "station_name"]],
    how="left",
    on=["p_picks", "pga_time"],
)

single_event_traces_info = traces_info[traces_info["EQ_ID"] == EQ_ID].drop_duplicates(
    subset=["station_name"]
)
event_prediction = pd.merge(
    event_prediction,
    single_event_traces_info[["station_name", "epdis (km)"]],
    how="left",
    on=["station_name"],
)

event = catalog[catalog["EQ_ID"] == EQ_ID]
event = event.assign(
    latitude=event["lat"] + event["lat_minute"] / 60,
    longitude=event["lon"] + event["lon_minute"] / 60,
)

fig, ax = plot_pga_map(
    trace_info=event_prediction,
    eventmeta=event,
    true_pga=event_prediction["answer"],
    pred_pga=event_prediction["predict"],
    sec=mask_after_sec,
    EQ_ID=EQ_ID,
    grid_method="linear",
    pad=100,
)
# fig.savefig(f"{path}/precision and warning time map/eq_id {EQ_ID}/{mask_after_sec} sec intensity map.png",
#             dpi=300)

fig, ax = warning_map(
    trace_info=event_prediction,
    eventmeta=event,
    EQ_ID=EQ_ID,
    sec=mask_after_sec,
    pga_threshold=pga_threshold,
)

# fig.savefig(f"{path}/precision and warning time map/eq_id {EQ_ID}/{mask_after_sec} sec warning map.png",
#             dpi=300)

fig, ax = correct_warning_with_epidist(
    event_prediction=event_prediction, mask_after_sec=mask_after_sec
)

# fig.savefig(f"{path}/precision and warning time map/eq_id {EQ_ID}/{mask_after_sec} sec epidist vs time.png",
#             dpi=300)

fig, ax = warning_time_hist(
    prediction,
    catalog,
    EQ_ID=EQ_ID,
    mask_after_sec=mask_after_sec,
    warning_mag_threshold=4,
)
# fig.savefig(f"{path}/precision and warning time map/eq_id {EQ_ID}/{mask_after_sec} sec warning stations hist.png",
#             dpi=300)

fig, ax = true_predicted(
    y_true=prediction[prediction["EQ_ID"] == EQ_ID]["answer"],
    y_pred=prediction[prediction["EQ_ID"] == EQ_ID]["predict"],
    time=mask_after_sec,
    quantile=False,
    agg="point",
    point_size=70,
)
true_predict_filter = (prediction["predict"] > pga_threshold) & (
    (prediction["answer"] > pga_threshold)
)
eq_id_filter = prediction["EQ_ID"] == EQ_ID
ax.scatter(
    prediction[eq_id_filter & true_predict_filter]["answer"],
    prediction[eq_id_filter & true_predict_filter]["predict"],
    s=70,
    c="red",
)
# fig.savefig(f"{path}/precision and warning time map/eq_id {EQ_ID}/{mask_after_sec} sec true vs predict.png",
#             dpi=300)
