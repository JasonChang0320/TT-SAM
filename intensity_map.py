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
label_type = "pga"
if label_type == "pga":
    label_threshold = np.log10(0.25)
    intensity = "IV"
if label_type == "pgv":
    label_threshold = np.log10(0.15)
    intensity = "V"
path = "./predict"
Afile_path = "data preprocess/events_traces_catalog"

catalog = pd.read_csv(f"{Afile_path}/1999_2019_final_catalog.csv")
traces_info = pd.read_csv(
    f"{Afile_path}/2009_2019_picked_traces_p_arrival_abstime_labeled_nostaoverlap.csv"
)
prediction = pd.read_csv(f"{path}/{mask_after_sec} sec ensemble (origin & big event model).csv")
station_info = pd.read_csv("data/station information/TSMIPstations_new.csv")

data_path = "D:/TEAM_TSMIP/data/TSMIP_1999_2019.hdf5"
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
# event_prediction.to_csv(
#     f"{path}/{mask_after_sec}_sec_meinong_eq_record_prediction.csv", index=False
# )

event = catalog[catalog["EQ_ID"] == EQ_ID]
event = event.assign(
    latitude=event["lat"] + event["lat_minute"] / 60,
    longitude=event["lon"] + event["lon_minute"] / 60,
)
# event.to_csv(f"{path}/meinong_eq_info.csv",index=False)

fig, ax = plot_intensity_map(
    trace_info=event_prediction,
    eventmeta=event,
    label_type=label_type,
    true_label=event_prediction["answer"],
    pred_label=event_prediction["predict"],
    sec=mask_after_sec,
    EQ_ID=EQ_ID,
    grid_method="linear",
    pad=100,
    title="2016 Meinong Earthquake PGA intensity Map",
)
# fig.savefig(f"{path}/{mask_after_sec} sec intensity map.png",
#             dpi=450,bbox_inches='tight')

fig, ax = warning_map(
    trace_info=event_prediction,
    eventmeta=event,
    label_type=label_type,
    intensity=intensity,
    EQ_ID=EQ_ID,
    sec=mask_after_sec,
    label_threshold=label_threshold,
)

# fig.savefig(f"{path}/meinong earthquake/{mask_after_sec} sec warning map.png",
#             dpi=300)
fig, ax = correct_warning_with_epidist(
    event_prediction=event_prediction,
    label_threshold=label_threshold,
    label_type=label_type,
    mask_after_sec=mask_after_sec,
)
# fig.savefig(f"{path}/meinong earthquake/{mask_after_sec} sec epidist vs time.png",
#             dpi=300)
fig, ax = warning_time_hist(
    prediction,
    catalog,
    EQ_ID=EQ_ID,
    mask_after_sec=mask_after_sec,
    warning_mag_threshold=4,
    label_threshold=label_threshold,
    label_type=label_type,
    bins=14,
)
# fig.savefig(
#     f"{path}/{mask_after_sec} sec warning stations hist.pdf",
#     dpi=300,
#     bbox_inches="tight",
# )

true_correct_filter = (prediction["predict"] > label_threshold) & (
    (prediction["answer"] > label_threshold)
)
false_correct_filter = (prediction["predict"] <= label_threshold) & (
    (prediction["answer"] <= label_threshold)
)
prediction[true_correct_filter | false_correct_filter]
eq_id_filter = prediction["EQ_ID"] == EQ_ID
fig, ax = true_predicted(
    y_true=prediction["answer"],
    y_pred=prediction["predict"],
    time=mask_after_sec,
    quantile=False,
    agg="point",
    point_size=70,
    target=label_type,
)
# ax.scatter(
#     prediction[eq_id_filter & true_predict_filter]["answer"],
#     prediction[eq_id_filter & true_predict_filter]["predict"],
#     s=70,
#     c="red",
#     alpha=0.5,
# )
# fig.savefig(f"{path}/meinong earthquake/{mask_after_sec} sec true vs predict.png",
#             dpi=300)


############################################################## M3.5 + M5.5
import matplotlib.pyplot as plt
import seaborn as sns

before_catalog = pd.read_csv(f"{Afile_path}/2009_2019_ok_events_p_arrival_abstime.csv")
after_catalog = pd.read_csv(f"{Afile_path}/1999_2019_final_catalog.csv")

before_catalog["from"] = "2009~2019 M>=3.5"
after_catalog["from"] = "1999~2008 M>=5.5"

catalog = pd.concat([before_catalog, after_catalog])
catalog.reset_index(inplace=True, drop=True)

fig, ax = plt.subplots(figsize=(7, 7))
sns.histplot(catalog, x="magnitude", hue="from", alpha=1, ax=ax)
ax.set_title("Events Catalog", fontsize=20)
ax.set_yscale("log")
ax.set_xlabel("Magnitude", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
###### trace 
before_trace = pd.read_csv(
    f"{Afile_path}/2009_2019_picked_traces_p_arrival_abstime_labeled_nostaoverlap.csv"
)
after_trace = pd.read_csv(f"{Afile_path}/1999_2019_final_traces.csv")

before_trace["from"] = "2009~2019 M>=3.5"
after_trace["from"] = "1999~2008 M>=5.5"

trace = pd.concat([before_trace, after_trace])
trace.reset_index(inplace=True, drop=True)
label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
pga_threshold = np.log10([0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10])
fig, ax = plt.subplots(figsize=(7, 7))
sns.histplot(trace, x="pga", hue="from", alpha=1, ax=ax, bins=32)
for i in range(len(pga_threshold) - 1):
    ax.text((pga_threshold[i] + pga_threshold[i + 1]) / 2, 10000, label[i])
ax.vlines(pga_threshold[1:-1], 0, 40000, linestyles="dotted", color="k")
ax.set_title("Traces catalog", fontsize=20)
ax.set_yscale("log")
ax.set_xlabel("PGA log(m/s^2)", fontsize=13)
ax.set_ylabel("Count", fontsize=13)

print(len(before_trace.query(f"pga >={pga_threshold[2]}")) / len(before_trace))
print(len(after_trace.query(f"pga >={pga_threshold[2]}")) / len(after_trace))
###### undersample
catalog=pd.read_csv(f"{Afile_path}/1999_2019_final_catalog.csv")
traces=pd.merge(after_trace,catalog,on="EQ_ID",how="left")
trace_num=traces["EQ_ID"].value_counts()
catalog["trace_num"]=catalog["EQ_ID"].map(trace_num)

drop_event=catalog.query("magnitude < 4 & trace_num < 25")

under_sampled_catalog=pd.concat([catalog,drop_event])
under_sampled_catalog.drop_duplicates(subset="EQ_ID",keep=False,inplace=True)
undersampled_trace=after_trace[after_trace["EQ_ID"].isin(under_sampled_catalog["EQ_ID"])]


# drop_event=catalog[catalog["magnitude"]<4.5].sample(frac=0.5)
# under_sampled_catalog=pd.concat([catalog,drop_event])
# under_sampled_catalog.drop_duplicates(subset="EQ_ID",keep=False,inplace=True)
# undersampled_trace=after_trace[after_trace["EQ_ID"].isin(under_sampled_catalog["EQ_ID"])]

print(len(traces.query(f"pga >={pga_threshold[2]}")) / len(after_trace))
print(len(undersampled_trace.query(f"pga >={pga_threshold[2]}")) / len(undersampled_trace))