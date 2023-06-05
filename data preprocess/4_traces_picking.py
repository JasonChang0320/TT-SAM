import pandas as pd
import sys
import os

sys.path.append("..")
from read_tsmip import read_tsmip
from obspy.signal.trigger import ar_pick

waveform_path = "../data/waveform"
traces = pd.read_csv("./events_traces_catalog/2009_2019_ok_traces.csv")

traces["p_pick_sec"] = 0
for i in range(len(traces)):
    print(f"{i}/{len(traces)}")
    EQ_ID = str(traces["EQ_ID"][i])
    year = str(traces["year"][i])
    month = str(traces["month"][i])
    day = str(traces["day"][i])
    hour = str(traces["hour"][i])
    minute = str(traces["minute"][i])
    second = str(traces["second"][i])
    intensity = str(traces["intensity"][i])
    station_name = traces["station_name"][i]
    epdis = str(traces["epdis (km)"][i])
    file_name = traces["file_name"][i].strip()
    if len(month) < 2:
        month = "0" + month
    waveform = read_tsmip(f"{waveform_path}/{year}/{month}/{file_name}.txt")
    # picking
    p_pick, _ = ar_pick(
        waveform[0],
        waveform[1],
        waveform[2],
        samp_rate=waveform[0].stats.sampling_rate,
        f1=1,  # Frequency of the lower bandpass window
        f2=20,  # Frequency of the upper bandpass window
        lta_p=1,  # Length of LTA for the P arrival in seconds
        sta_p=0.1,  # Length of STA for the P arrival in seconds
        lta_s=4.0,  # Length of LTA for the S arrival in seconds
        sta_s=1.0,  # Length of STA for the P arrival in seconds
        m_p=2,  # Number of AR coefficients for the P arrival
        m_s=8,  # Number of AR coefficients for the S arrival
        l_p=0.1,
        l_s=0.2,
        s_pick=False,
    )
    traces.loc[i, "p_pick_sec"] = p_pick

# traces.to_csv("events_traces_catalog/2009_2019_ok_picked_traces.csv",index=False)

# ========shift p_picking by velocity model to correct absolute time======
traces = pd.read_csv("events_traces_catalog/2009_2019_ok_picked_traces.csv")

EQ_ID = os.listdir("./tracer_demo/2009_2019_output")

traces["p_arrival_abs_time"] = pd.to_datetime(
    traces[["year", "month", "day", "hour", "minute", "second"]]
)

colnames = [
    "evt_lon",
    "evt_lat",
    "evt_depth",
    "sta_lon",
    "sta_lat",
    "sta_elev",
    "p_arrival",
    "s_arrival",
]
for eq in EQ_ID:
    event_file_path = f"./tracer_demo/2009_2019_output/{eq}/output.table"
    tracer_output = pd.read_csv(
        event_file_path, sep=r"\s+", names=colnames, header=None
    )
    trace_index = traces[traces["EQ_ID"] == int(eq)].index
    p_arrival = pd.to_timedelta(tracer_output["p_arrival"], unit="s")
    p_arrival.index = trace_index
    traces.loc[trace_index, "p_arrival_abs_time"] = (
        traces.loc[trace_index, "p_arrival_abs_time"] + p_arrival
    )
# traces 和 event 須將 eq_id: 29363 剔除 (velocity model calculate out of range)
final_traces = traces[traces["EQ_ID"] != 29363]

event = pd.read_csv("./events_traces_catalog/2009_2019_ok_events.csv")
final_event = event[event["EQ_ID"] != 29363]
# 存檔
final_traces.to_csv(
    "./events_traces_catalog/2009_2019_picked_traces_p_arrival_abstime.csv", index=False
)
final_event.to_csv(
    "./events_traces_catalog/2009_2019_ok_events_p_arrival_abstime.csv", index=False
)
