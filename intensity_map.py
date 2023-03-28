import h5py
import numpy as np
import pandas as pd

from plot_predict_map import (
    correct_warning_with_epidist,
    plot_intensity_map,
    true_predicted,
    warning_map,
    warning_time_hist,
)

mask_after_sec = 5
EQ_ID = 24784
trigger_station_threshold = 1
label_type = "pgv"
if label_type == "pga":
    label_threshold = np.log10(9.8 * 0.025)
    intensity = "IV"
if label_type == "pgv":
    label_threshold = np.log10(0.15)
    intensity = "V"
path = "./predict/dis random sec predict pgv test 2016/ok model prediction"
Afile_path = "data/Afile"

catalog = pd.read_csv(f"{Afile_path}/final catalog (station exist)_filtered.csv")
traces_info = pd.read_csv(
    f"{Afile_path}/1991-2020 traces with picking and label_new (sta location exist)_filtered.csv"
)
prediction = pd.read_csv(f"{path}/model8 12 {mask_after_sec} sec prediction.csv")
station_info = pd.read_csv("data/station information/TSMIPstations_new.csv")

data_path = "D:/TEAM_TSMIP/data/TSMIP_filtered.hdf5"
dataset = h5py.File(data_path, "r")
p_picks = dataset["data"][str(EQ_ID)]["p_picks"][:].tolist()
label_time = dataset["data"][str(EQ_ID)][f"{label_type}_time"][:].tolist()
station_name = dataset["data"][str(EQ_ID)]["station_name"][:].tolist()
latitude = dataset["data"][str(EQ_ID)]["station_location"][:, 0].tolist()
longitude = dataset["data"][str(EQ_ID)]["station_location"][:, 1].tolist()
elevation = dataset["data"][str(EQ_ID)]["station_location"][:, 2].tolist()
label = dataset["data"][str(EQ_ID)][f"{label_type}"][:].tolist()

station_df = pd.DataFrame(
    {
        "p_picks": p_picks,
        f"{label_type}": label,
        f"{label_type}_time": label_time,
        "station_name": station_name,
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
    }
)
dataset.close()

station_df["station_name"] = station_df["station_name"].str.decode("utf-8")
station_df = station_df.drop_duplicates(
    subset=["p_picks", f"{label_type}_time", "station_name"], keep="last"
)

event_prediction = prediction[prediction["EQ_ID"] == EQ_ID]
event_prediction = event_prediction.drop_duplicates(
    subset=["p_picks", f"{label_type}_time", "predict"], keep="last"
)

event_prediction = pd.merge(
    event_prediction,
    station_df[["p_picks", f"{label_type}_time", "station_name"]],
    how="left",
    on=["p_picks", f"{label_type}_time"],
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

fig, ax = plot_intensity_map(
    trace_info=event_prediction,
    eventmeta=event,
    label_type="pgv",
    true_label=event_prediction["answer"],
    pred_label=event_prediction["predict"],
    sec=mask_after_sec,
    EQ_ID=EQ_ID,
    grid_method="linear",
    pad=100,
)
# fig.savefig(f"{path}/precision and warning time map/EQID_{EQ_ID}/{mask_after_sec} sec intensity map.png",
#             dpi=300)

fig, ax = warning_map(
    trace_info=event_prediction,
    eventmeta=event,
    label_type="pgv",
    intensity=intensity,
    EQ_ID=EQ_ID,
    sec=mask_after_sec,
    label_threshold=label_threshold,
)

# fig.savefig(f"{path}/precision and warning time map/EQID_{EQ_ID}/{mask_after_sec} sec warning map.png",
#             dpi=300)

fig, ax = correct_warning_with_epidist(
    event_prediction=event_prediction,
    label_threshold=label_threshold,
    label_type=label_type,
    mask_after_sec=mask_after_sec,
)

# fig.savefig(f"{path}/precision and warning time map/EQID_{EQ_ID}/{mask_after_sec} sec epidist vs time.png",
#             dpi=300)

fig, ax = warning_time_hist(
    prediction,
    catalog,
    EQ_ID=EQ_ID,
    mask_after_sec=mask_after_sec,
    warning_mag_threshold=4,
    label_threshold=label_threshold,
    label_type="pgv",
)
# fig.savefig(f"{path}/precision and warning time map/EQID_{EQ_ID}/{mask_after_sec} sec warning stations hist.png",
#             dpi=300)

true_predict_filter = (prediction["predict"] > label_threshold) & (
    (prediction["answer"] > label_threshold)
)
eq_id_filter = prediction["EQ_ID"] == EQ_ID
fig, ax = true_predicted(
    y_true=prediction[eq_id_filter & ~true_predict_filter]["answer"],
    y_pred=prediction[eq_id_filter & ~true_predict_filter]["predict"],
    time=mask_after_sec,
    quantile=False,
    agg="point",
    point_size=70,
    target="pgv",
)
ax.scatter(
    prediction[eq_id_filter & true_predict_filter]["answer"],
    prediction[eq_id_filter & true_predict_filter]["predict"],
    s=70,
    c="red",
    alpha=0.5,
)
# fig.savefig(f"{path}/precision and warning time map/EQID_{EQ_ID}/{mask_after_sec} sec true vs predict.png",
#             dpi=300)
