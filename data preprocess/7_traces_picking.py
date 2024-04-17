import pandas as pd
import os

from read_tsmip import read_tsmip
from obspy.signal.trigger import ar_pick

start_year=1999
end_year=2008
waveform_path = "../data/waveform"
traces = pd.read_csv(f"./events_traces_catalog/{start_year}_{end_year}_ok_traces.csv")

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

# traces.to_csv(f"events_traces_catalog/{start_year}_{end_year}_ok_picked_traces.csv",index=False)


