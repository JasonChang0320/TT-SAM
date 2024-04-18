import pandas as pd
import numpy as np

from read_tsmip import read_tsmip, get_peak_value, get_integrated_stream

# read traces catalog
start_year=1999
end_year=2008
waveform_path = "../data/waveform"
traces = pd.read_csv(
    f"./events_traces_catalog/{start_year}_{end_year}_picked_traces_p_arrival_abstime.csv"
)

sampling_rate = 200
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
    # read waveform
    waveform = read_tsmip(f"{waveform_path}/{year}/{month}/{file_name}.txt")
    # resample to 200Hz
    if waveform[0].stats.sampling_rate != sampling_rate:
        waveform.resample(sampling_rate, window="hann")

    # detrend
    waveform.detrend(type="demean")
    # lowpass filter
    waveform.filter("lowpass", freq=10)  # filter
    # get pga
    pick_point = int(np.round(traces["p_pick_sec"][i] * sampling_rate, 0))
    pga, pga_time = get_peak_value(waveform, pick_point=pick_point)
    # waveform taper
    waveform.taper(max_percentage=0.05, type="cosine")
    # integrate
    vel_waveform = get_integrated_stream(waveform)
    # bandpass filter
    vel_waveform.filter("bandpass", freqmin=0.075, freqmax=10)
    # get pgv
    pgv, pgv_time = get_peak_value(vel_waveform, pick_point=pick_point)
    # input to df
    traces.loc[i, "pga"] = pga
    traces.loc[i, "pga_time"] = pga_time
    traces.loc[i, "pgv"] = pgv
    traces.loc[i, "pgv_time"] = pgv_time

# traces.to_csv(
#     f"./events_traces_catalog/{start_year}_{end_year}_picked_traces_p_arrival_abstime_labeled.csv",
#     index=False,
# )
