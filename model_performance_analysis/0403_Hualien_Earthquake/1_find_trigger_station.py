import pandas as pd
import os
import obspy
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.trigger import ar_pick
import json


def dist(event_lat, event_lon, station_lat, station_lon):  # unit: degree
    dist = ((event_lat - station_lat) ** 2 + (event_lon - station_lon) ** 2) ** (1 / 2)
    return dist


mask_after_sec = 5
station_info = pd.read_csv("../../data/station_information/TSMIPstations_new.csv")
traces_info_with_vs30 = pd.read_csv(
    "../../data_preprocess/events_traces_catalog/1999_2019_final_traces_Vs30.csv"
)
sample_rate = 200

path = "./113019_TSMIP_SAC"
waveform_files = os.listdir(path)

stations = []
for file in waveform_files:
    station_name = file[:6]
    if station_name not in stations:
        stations.append(station_name)

station_info = station_info[station_info["location_code"].isin(stations)]
station_info = station_info.reset_index(drop=True)

# event epicenter
event_lat = 23.77
event_lon = 121.67

dist_dict = {"dist": []}
for i in range(len(station_info)):
    station_lat = station_info["latitude"][i]
    station_lon = station_info["longitude"][i]
    dist_dict["dist"].append(dist(event_lat, event_lon, station_lat, station_lon))
station_info["dist (degree)"] = dist_dict["dist"]

station_info["p_picks (sec)"] = 0
check_station = ["HWA026", "HWA067", "HWA025", "TTN032", "ILA050"]
# plot and picking:
for i, station in enumerate(station_info["location_code"]):
    trace_z = obspy.read(f"{path}/{station}.Z.SAC")
    trace_n = obspy.read(f"{path}/{station}.N.SAC")
    trace_e = obspy.read(f"{path}/{station}.E.SAC")
    trace_z.resample(sample_rate, window="hann")
    trace_n.resample(sample_rate, window="hann")
    trace_e.resample(sample_rate, window="hann")

    waveforms = np.array([trace_z[0].data, trace_n[0].data, trace_e[0].data])
    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(waveforms[0])
    # ax[1].plot(waveforms[1])
    # ax[2].plot(waveforms[2])
    # ax[0].set_title(
    #     f"{station}_{trace_z[0].stats.starttime}-{trace_z[0].stats.endtime}"
    # )
    try:
        p_pick, _ = ar_pick(
            waveforms[0],
            waveforms[1],
            waveforms[2],
            samp_rate=200,
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
        station_info.loc[i, "p_picks (sec)"] = p_pick
        # ax[0].axvline(x=p_pick * sample_rate, color="r", linestyle="-")
        # ax[1].axvline(x=p_pick * sample_rate, color="r", linestyle="-")
        # ax[2].axvline(x=p_pick * sample_rate, color="r", linestyle="-")
    except:
        station_info.loc[i, "p_picks (sec)"] = p_pick
    # fig.savefig(f"0403waveform_image/{station}.png", dpi=300)
    plt.close()

station_info = station_info.sort_values(by="dist (degree)")
station_info = station_info.reset_index(drop=True)

trigger_station_info = pd.merge(
    station_info,
    traces_info_with_vs30[["station_name", "Vs30"]].drop_duplicates(
        subset="station_name"
    ),
    left_on="location_code",
    right_on="station_name",
    how="left",
)
trigger_station_info = trigger_station_info.dropna(
    subset=["latitude", "longitude", "elevation (m)", "Vs30"]
)
trigger_station_info=trigger_station_info[trigger_station_info["station_name"]!="HWA026"]
trigger_station_info=trigger_station_info[trigger_station_info["station_name"]!="HWA067"]
trigger_station_info=trigger_station_info[trigger_station_info["station_name"]!="HWA025"]
trigger_station_info=trigger_station_info[trigger_station_info["station_name"]!="ILA050"]
trigger_station_info = trigger_station_info.reset_index(drop=True)

P_wave_velocity = 6.5
stream = obspy.core.stream.Stream()
waveforms_window = []
mask_station_index = []
target_length = 18000
for i, station in enumerate(trigger_station_info["location_code"][:25]):
    trace_z = obspy.read(f"{path}/{station}.Z.SAC")
    trace_n = obspy.read(f"{path}/{station}.N.SAC")
    trace_e = obspy.read(f"{path}/{station}.E.SAC")
    # bad data padding to fit time window
    # HWA026 HWA067 HWA025 ILA050
    for trace in [trace_z, trace_n, trace_e]:
        trace[0].data = trace[0].data/100 #cm/s2 to m/s2
        if len(trace[0].data) < target_length:
            padding_length = target_length - len(trace[0].data)
            padding = np.zeros(padding_length)
            trace[0].data = np.concatenate((trace[0].data, padding))
    trace_z.resample(200, window="hann")
    trace_n.resample(200, window="hann")
    trace_e.resample(200, window="hann")

    waveforms = np.array([trace_z[0].data, trace_n[0].data, trace_e[0].data])
    if station == "HWA074":  # first triggered station
        p_pick, _ = ar_pick(
            waveforms[0],
            waveforms[1],
            waveforms[2],
            samp_rate=200,
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
    start_time = int((p_pick - 5) * sample_rate)
    end_time = int((p_pick + 10) * sample_rate)
    trace_z[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0
    trace_n[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0
    trace_e[0].data[int((p_pick) * sample_rate) + (mask_after_sec * sample_rate) :] = 0

    if (
        trigger_station_info["dist (degree)"][i]
        - trigger_station_info["dist (degree)"][0]
    ) * 100 / P_wave_velocity > mask_after_sec:  # zero padding non triggered station
        mask_station_index.append(i) #for mask non trigger station information
        trace_z[0].data[:] = 0
        trace_n[0].data[:] = 0
        trace_e[0].data[:] = 0
    waveforms = np.stack(
        (
            trace_z[0].data[start_time:end_time],
            trace_n[0].data[start_time:end_time],
            trace_e[0].data[start_time:end_time],
        ),
        axis=1,
    )
    waveforms = waveforms.reshape(3000, 3)
    waveforms_window.append(waveforms)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(waveforms[:, 0])
    ax[1].plot(waveforms[:, 1])
    ax[2].plot(waveforms[:, 2])
    ax[0].set_title(f"{station}")
    # plt.close()
    # fig.savefig(
    #     f"model_input_waveform_image/{mask_after_sec}_sec/{i}_{station}.png", dpi=300
    # )

waveform = np.stack(waveforms_window, axis=0).tolist()
target_station_info = trigger_station_info.copy()

#mask non trigger station information
for i in mask_station_index:
    trigger_station_info.loc[i, ["latitude", "longitude", "elevation (m)", "Vs30"]] = 0

input_station = (
    trigger_station_info[["latitude", "longitude", "elevation (m)", "Vs30"]][:25]
    .to_numpy()
    .tolist()
)
for i in range(1, 16):
    print((i - 1) * 25, i * 25)
    target_station = (
        target_station_info[["latitude", "longitude", "elevation (m)", "Vs30"]][
            (i - 1) * 25 : i * 25
        ]
        .to_numpy()
        .tolist()
    )
    station_name = target_station_info["location_code"][(i - 1) * 25 : i * 25].tolist()
    output = {
        "waveform": waveform,
        "sta": input_station,
        "target": target_station,
        "station_name": station_name,
    }

    # with open(f"model_input/{mask_after_sec}_sec_without_broken_data/{i}.json", "w") as json_file:
    #     json.dump(output, json_file)
