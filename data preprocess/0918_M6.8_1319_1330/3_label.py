import pandas as pd
import numpy as np
import sys
import os
import obspy
import re
from obspy.geodetics import gps2dist_azimuth 

sys.path.append("../..")
from read_tsmip import read_tsmip, get_peak_value, get_integrated_stream

# read traces catalog
waveform_path = "../../data/0918_M6.8_1319_1330/ascii"
traces = pd.read_csv("traces_catalog.csv")
# delete broken waveform
traces = traces.query("quality_control=='y'").reset_index(drop=True)

sampling_rate = 200
for i in range(len(traces)):
    print(f"{i}/{len(traces)}")
    file_name = traces["file_name"][i].strip()
    # read waveform
    data = pd.read_csv(
        f"{waveform_path}/{file_name}.asc", sep="\s+", skiprows=1, header=None
    ).to_numpy()

    with open(f"{waveform_path}/{file_name}.asc", "r") as f:
        picks = f.readlines()[0]
        picks = re.findall(r"\d+\.\d+", picks)
        picks = [np.round(float(number), 2) for number in picks]

    waveform = obspy.core.stream.Stream()
    channel = ["HLZ", "HLN", "HLE"]
    for j, chan in enumerate(channel):
        start = np.where(data == picks[2])[0][0]
        end = np.where(data == picks[3])[0][0]
        trace = obspy.core.trace.Trace(data[start:end, j + 1])

        trace.stats.network = "TW"
        # trace.stats.station = header[0][14:20]
        trace.stats.channel = chan

        trace.stats.sampling_rate = int(1 / abs(data[0, 0] - data[1, 0]))

        waveform.append(trace)
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


#calculate epicentral distance

catalog = pd.read_csv("event_catalog.csv")
traces["epdis (km)"]=0

eq_latitude = catalog["lat"][0] + catalog["lat_minute"][0] / 60
eq_longitude = catalog["lon"][0] + catalog["lon_minute"][0] / 60
eq_depth = catalog["depth"][0]
for i in range(len(traces)):
    station_latitude = traces["latitude"][i]
    station_longitude = traces["longitude"][i]
    station_elevation = traces["elevation (m)"][i] / 1000
    epi_dis, azimuth, _ = gps2dist_azimuth(
        eq_latitude, eq_longitude, station_latitude, station_longitude
    )
    epi_dis=(epi_dis**2 + (eq_depth - station_elevation)**2)**0.5
    traces.loc[i,"epdis (km)"]=epi_dis/1000

traces.to_csv(f"traces_catalog.csv", index=False)