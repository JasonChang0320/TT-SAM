import pandas as pd
from tqdm import tqdm

import obspy
import matplotlib.pyplot as plt
from read_tsmip import cut_traces

start_year = 1999
end_year = 2019
Afile_path = "data/Afile"
sta_path = "../data/station_information"
waveform_path = "../data/waveform"
catalog = pd.read_csv(
    f"./events_traces_catalog/{start_year}_{end_year}_final_catalog.csv"
)
traces = pd.read_csv(
    f"./events_traces_catalog/{start_year}_{end_year}_final_traces_Vs30.csv"
)
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv") 
traces.loc[traces.index, "p_pick_sec"] = pd.to_timedelta(
    traces["p_pick_sec"], unit="sec"
)
traces.loc[traces.index, "p_arrival_abs_time"] = pd.to_datetime(
    traces["p_arrival_abs_time"], format="%Y-%m-%d %H:%M:%S"
)

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
        # fig.savefig(f"cut event figure/{eq_id}_{trace.stats.channel}.png",dpi=300)
        plt.close()