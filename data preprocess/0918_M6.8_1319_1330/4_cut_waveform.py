import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import obspy
import matplotlib.pyplot as plt

sys.path.append("../..")
from read_tsmip import cut_traces

sta_path = "../../data/station_information"
waveform_path = "../../data/0918_M6.8_1319_1330/ascii"
traces = pd.read_csv("traces_catalog.csv")
catalog = pd.read_csv("event_catalog.csv")
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv") 
traces.loc[traces.index, "p_pick_sec"] = pd.to_timedelta(
    traces["p_pick_sec"], unit="sec"
)
traces.loc[traces.index, "p_arrival_abs_time"] = pd.to_datetime(
    traces["p_arrival_abs_time"], format="%Y-%m-%d %H:%M:%S"
)

output = f"../../data/TSMIP_0918_M6.8_1319_1330.hdf5"
error_event = {"EQ_ID": [], "reason": []}
with h5py.File(output, "w") as file:
    data = file.create_group("data")
    meta = file.create_group("metadata")
    for eq_id in tqdm(catalog["EQ_ID"]):
        # for eq_id in [247]:
        # try:
        tmp_traces, traces_info = cut_traces(
            traces, eq_id, waveform_path, waveform_type="acc"
        )
        _, vel_info = cut_traces(traces, eq_id, waveform_path, waveform_type="vel")
        _, dis_info = cut_traces(traces, eq_id, waveform_path, waveform_type="dis")
        traces_info["vel"] = vel_info["traces"]
        traces_info["dis"] = dis_info["traces"]
        # fig=plot_cutting_event(tmp_traces,traces_info)
        start_time_str_arr = np.array(traces_info["start_time"], dtype="S30")
        station_name_str_arr = np.array(tmp_traces["station_name"], dtype="S30")
        tmp_station_info = pd.merge(
            tmp_traces[["station_name","Vs30"]],
            station_info[
                ["location_code", "latitude", "longitude", "elevation (m)"]
            ],
            how="left",
            left_on="station_name",
            right_on="location_code",
        )
        location_array = np.array(
            tmp_station_info[["latitude", "longitude", "elevation (m)"]]
        )
        Vs30_array=np.array(tmp_traces["Vs30"])
        if np.isnan(location_array).any():
            print("The location array contain NaN values")
            continue
        event = data.create_group(f"{eq_id}")
        event.create_dataset(
            "acc_traces", data=traces_info["traces"], dtype=np.float64
        )
        event.create_dataset(
            "vel_traces", data=traces_info["vel"], dtype=np.float64
        )
        event.create_dataset(
            "dis_traces", data=traces_info["dis"], dtype=np.float64
        )
        event.create_dataset("p_picks", data=traces_info["p_picks"], dtype=np.int64)
        event.create_dataset("pga", data=traces_info["pga"], dtype=np.float64)
        event.create_dataset("pgv", data=traces_info["pgv"], dtype=np.float64)
        event.create_dataset(
            "start_time", data=start_time_str_arr, maxshape=(None), chunks=True
        )
        event.create_dataset(
            "pga_time", data=traces_info["pga_time"], dtype=np.int64
        )
        event.create_dataset(
            "pgv_time", data=traces_info["pgv_time"], dtype=np.int64
        )
        event.create_dataset(
            "station_name", data=station_name_str_arr, maxshape=(None), chunks=True
        )
        event.create_dataset(
            "station_location", data=location_array, dtype=np.float64
        )
        event.create_dataset(
            "Vs30", data=Vs30_array, dtype=np.float64
        )
        # except Exception as reason:
        #     print(f"EQ_ID:{eq_id}, {reason}")
        #     error_event["EQ_ID"].append(eq_id)
        #     error_event["reason"].append(reason)
        #     continue
        # fig.savefig(f"data/cutting waveform image/{eq_id}.png")
    error_event_df = pd.DataFrame(error_event)
    error_event_df.to_csv(
        "./load into hdf5 error event.csv", index=False
    )

catalog.to_hdf(output, key="metadata/event_metadata", mode="a", format="table")
traces.to_hdf(output, key="metadata/traces_metadata", mode="a", format="table")

# plot records section
for eq_id in tqdm(catalog["EQ_ID"]):
    tmp_traces, traces_info = cut_traces(traces, eq_id, waveform_path, waveform_type="acc")
    for i,chan in enumerate(["HLZ","HLN","HLE"]):
        stream = obspy.core.stream.Stream()
        for j in range(len(traces_info["traces"])):
            trace = obspy.core.trace.Trace(data=traces_info["traces"][j][:, i])
            trace.stats.id = eq_id
            trace.stats.station = tmp_traces["station_name"][j]
            trace.stats.channel = chan
            trace.stats.distance = tmp_traces["epdis (km)"][j] * 1000
            trace.stats.starttime = traces_info["start_time"][j]
            trace.stats.sampling_rate = 200

            stream.append(trace)
        fig, ax = plt.subplots()
        stream.plot(type="section",fig=fig)

        magnitude = catalog[catalog["EQ_ID"] == eq_id]["magnitude"].values[0]

        ax.set_title(
            f"EQ ID:{eq_id}, Magnitude: {magnitude}, start time: {traces_info['start_time'][j]}"
        )
        fig.savefig(f"cut trace/{eq_id}_{trace.stats.channel}.png")
        plt.close()

for i in range(len(traces_info["traces"])):
    file_name=tmp_traces["file_name"][i]
    station_name=tmp_traces["station_name"][i]
    p_pick=traces_info["p_picks"][i]
    fig,ax=plt.subplots(3,1,figsize=(14,7))
    for j in range(len(ax)):
        ax[j].plot(traces_info["traces"][i][:,j])
        ax[j].axvline(x=p_pick, color="r", linestyle="-")
    ax[0].set_title(f"EQ_ID:{eq_id},station_name: {station_name},cut from file_name:{file_name}")
    fig.savefig(f"cut trace/EQ_ID_{eq_id}_{station_name}.png",dpi=300)
    plt.close()