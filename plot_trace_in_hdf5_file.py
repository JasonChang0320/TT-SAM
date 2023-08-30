import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

year = 2018
month = 2
output_path=f"data/waveform_in_hdf5/{year}_{month}"
if not os.path.exists(f"{output_path}"):
    os.makedirs(f"{output_path}")
mag_threshold = 3.5
input_types = ["acc", "vel", "dis"]
label_key = "pga"
data_path = "data/TSMIP_2009_2019.hdf5"
event_metadata = pd.read_hdf(data_path, "metadata/event_metadata")
trace_metadata = pd.read_hdf(data_path, "metadata/traces_metadata")
tmp_event = event_metadata.query(
    f"year == {year} & month == {month} & magnitude>={mag_threshold}"
)
tmp_traces = trace_metadata[trace_metadata["EQ_ID"].isin(tmp_event["EQ_ID"].values)]
# for _, event in tmp_event.iterrows():
#     print(event)
with h5py.File(data_path, "r") as file:
    decimate = 1
    data = {"EQ_ID": []}
    for _, event in tmp_event.iterrows():
        event_name = str(int(event["EQ_ID"]))
        data["EQ_ID"].append(event_name)
        # print(event_name)
        g_event = file["data"][event_name]
        for key in g_event:
            if key not in data:
                data[key] = []
            data[key] += [g_event[key][()]]
    for i in range(len(data["EQ_ID"])):
        EQ_ID = data["EQ_ID"][i]
        traces_info = tmp_traces.query(f"EQ_ID == {EQ_ID}")
        info = pd.merge(traces_info, tmp_event, how="left", on="EQ_ID")
        for input_type,unit in zip(input_types,["cm/s^2","cm/s","cm"]):
            for j in range(data[f"{input_type}_traces"][i].shape[0]):  # record 數量
                fig, ax = plt.subplots(data[f"{input_type}_traces"][i].shape[2], 1, figsize=(14, 7))
                for k in range(data[f"{input_type}_traces"][i].shape[2]):  # component
                    ax[k].plot(data[f"{input_type}_traces"][i][j, :, k] * 100)
                    ax[k].axvline(x=data["p_picks"][i][j], c="r")
                magnitude = info["magnitude"][j]
                station_name = info["station_name"][j]
                epdis = info["epdis (km)"][j]
                ax[0].set_title(
                    f"EQ_ID:{EQ_ID}, station: {station_name}, magnitude:{magnitude}, epdis: {epdis} (km)"
                )
                ax[1].set_ylabel(f"{unit}")
                # plt.close()
                # fig.savefig(f"{output_path}/EQID_{EQ_ID}_{station_name}_{input_type}.png")
        break
