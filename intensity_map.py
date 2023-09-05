import h5py
import numpy as np
import pandas as pd
import os
from plot_predict_map import (
    correct_warning_with_epidist,
    plot_intensity_map,
    true_predicted,
    warning_map,
    warning_time_hist,
)

mask_after_sec = 5
label_type = "pga"
if label_type == "pga":
    label_threshold = np.log10(0.25)
    intensity = "IV"
if label_type == "pgv":
    label_threshold = np.log10(0.15)
    intensity = "V"

path = "./predict/station_blind_Vs30_bias2closed_station_2016"
output_path=f"{path}/mag bigger 5.5 predict"
if not os.path.isdir(output_path):
    os.mkdir(output_path)
Afile_path = "data preprocess/events_traces_catalog"

catalog = pd.read_csv(f"{Afile_path}/1999_2019_final_catalog.csv")
traces_info = pd.read_csv(
    f"{Afile_path}/2009_2019_picked_traces_p_arrival_abstime_labeled_nostaoverlap.csv"
)
prediction_with_info = pd.read_csv(
    f"{path}/{mask_after_sec} sec model11 with all info.csv"
)

for EQ_ID in catalog.query("year==2016 & magnitude>=5.5")["EQ_ID"]:
    event = catalog[catalog["EQ_ID"] == EQ_ID]
    event = event.assign(
        latitude=event["lat"] + event["lat_minute"] / 60,
        longitude=event["lon"] + event["lon_minute"] / 60,
    )
    event_prediction = prediction_with_info.query(f"EQ_ID=={EQ_ID}")

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
        title=f"EQID: {EQ_ID}, mag: {event['magnitude'].values[0]}, {mask_after_sec} sec PGA intensity Map",
    )
    # fig.savefig(
    #     f"{output_path}/{EQ_ID}_mag_{event['magnitude'].values[0]}_{mask_after_sec}sec PGA intensity Map.png", dpi=450, bbox_inches="tight"
    # )
    fig, ax = true_predicted(
        y_true=event_prediction["answer"],
        y_pred=event_prediction["predict"],
        time=mask_after_sec,
        quantile=False,
        agg="point",
        point_size=70,
        target=label_type,
        title=f"EQID: {EQ_ID}, mag: {event['magnitude'].values[0]}, {mask_after_sec} sec true and predict",
    )
    # fig.savefig(
    #     f"{output_path}/{EQ_ID}_mag_{event['magnitude'].values[0]}_{mask_after_sec}sec true predict plot.png", dpi=450, bbox_inches="tight"
    # )
    try:
        fig, ax = warning_map(
            trace_info=event_prediction,
            eventmeta=event,
            label_type=label_type,
            intensity=intensity,
            EQ_ID=EQ_ID,
            sec=mask_after_sec,
            label_threshold=label_threshold,
        )

        # fig.savefig(f"{output_path}/{EQ_ID}_mag_{event['magnitude'].values[0]}_{mask_after_sec} sec warning map.png",
        #             dpi=300)
        fig, ax = correct_warning_with_epidist(
            event_prediction=event_prediction,
            label_threshold=label_threshold,
            label_type=label_type,
            mask_after_sec=mask_after_sec,
        )
        # fig.savefig(f"{output_path}/{EQ_ID}_mag_{event['magnitude'].values[0]}_{mask_after_sec} sec epidist vs time.png",
        #             dpi=300)
        fig, ax = warning_time_hist(
            event_prediction,
            catalog,
            EQ_ID=EQ_ID,
            mask_after_sec=mask_after_sec,
            warning_mag_threshold=4,
            label_threshold=label_threshold,
            label_type=label_type,
            bins=14,
        )
        # fig.savefig(
        #     f"{output_path}/{EQ_ID}_mag_{event['magnitude'].values[0]}_{mask_after_sec} sec warning stations hist.png",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
    except Exception as e:
        print(EQ_ID)
        continue



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
