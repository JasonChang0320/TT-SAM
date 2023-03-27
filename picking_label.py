import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from read_tsmip import *

Afile_path = "data/Afile"
traces = pd.read_csv(
    f"{Afile_path}/1991-2020 traces (no broken data, double event).csv"
)
sampling_rate = 200

add_columns = ["p_picks (sec)", "s_picks (sec)", "pga", "pga_time", "pgv", "pgv_time"]

traces[add_columns] = np.nan

error_file = {"year": [], "month": [], "file": [], "reason": []}

for trace_index in tqdm(traces.index):
    if trace_index == 53398:  # cause kernel crush
        continue
    year = traces["year"][trace_index]
    month = traces["month"][trace_index]
    path = f"data/waveform/{year}/{month}"
    file_name = traces["file_name"][trace_index]
    if len(str(month)) < 2:
        month = "0" + str(month)
    path = f"data/waveform/{year}/{month}"
    file_name = file_name.strip()
    try:
        trace = read_tsmip(f"{path}/{file_name}.txt")
        trace.detrend(type="linear")  # baseline correction
        trace.filter("lowpass", freq=10)  # filter
        if trace[0].stats.sampling_rate != sampling_rate:
            trace.resample(sampling_rate, window="hann")
        # picking
        p_pick, s_pick = ar_pick(
            trace[0],
            trace[1],
            trace[2],
            samp_rate=sampling_rate,
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
            s_pick=True,
        )
        # p_pick,s_pick,fig=trace_pick_plot(trace,file_name)
        # get pga
        pga, pga_times = get_peak_value(trace)  # label unit log(m/s2)
        # get pgv
        vel_stream = get_integrated_stream(trace, baseline_correction=True, filter=True)
        pgv, pgv_times = get_peak_value(vel_stream)  # label unit log(m/s)
        traces.loc[trace_index, "p_picks (sec)"] = p_pick
        traces.loc[trace_index, "s_picks (sec)"] = s_pick
        traces.loc[trace_index, "pga"] = pga
        traces.loc[trace_index, "pga_time"] = pga_times
        traces.loc[trace_index, "pgv"] = pgv
        traces.loc[trace_index, "pgv_time"] = pgv_times
    except Exception as reason:
        print(file_name, f"year:{year},month:{month}, {reason}")
        error_file["year"].append(year)
        error_file["month"].append(month)
        error_file["file"].append(file_name)
        error_file["reason"].append(reason)
        continue

error_file_df = pd.DataFrame(error_file)
error_file_df["reason"] = error_file_df["reason"].astype(str)
traces.to_csv(
    f"{Afile_path}/1991-2020 traces with picking and label filtered.csv", index=False
)
error_file_df.to_csv(
    f"{Afile_path}/1991-2020 error in picking and label_first filtered.csv", index=False
)

# picking again
traces = pd.read_csv(
    f"{Afile_path}/1991-2020 traces with picking and label filtered.csv"
)
error_file_df = pd.read_csv(
    f"{Afile_path}/1991-2020 error in picking and label filtered.csv"
)
traces["file_name"] = traces["file_name"].str.strip()
pick_again_filter = error_file_df["reason"].str.contains(r"^exception")

pick_again_df = error_file_df[pick_again_filter]

for i in pick_again_df.index:
    year = pick_again_df["year"][i]
    month = pick_again_df["month"][i]
    path = f"data/waveform/{year}/{month}"
    file_name = pick_again_df["file"][i]
    if len(str(month)) < 2:
        month = "0" + str(month)
    path = f"data/waveform/{year}/{month}"
    file_name = file_name.strip()
    trace = read_tsmip(f"{path}/{file_name}.txt")
    trace.detrend(type="linear")  # baseline correction
    trace.filter("lowpass", freq=10)  # filter
    # picking
    if trace[0].stats.sampling_rate != sampling_rate:
        trace.resample(sampling_rate, window="hann")
    try:
        p_pick, s_pick = ar_pick(
            trace[0],
            trace[1],
            trace[2],
            samp_rate=sampling_rate,
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
            s_pick=True,
        )
    except:
        print(i)
    # get pga
    pga, pga_times = get_peak_value(trace)
    # get pgv
    vel_stream = get_integrated_stream(trace, baseline_correction=True, filter=True)
    pgv, pgv_times = get_peak_value(vel_stream)

    insert_filter = (
        (traces["year"] == year)
        & (traces["month"] == int(month))
        & (traces["file_name"] == file_name)
    )
    trace_index = traces[insert_filter].index[0]
    traces.loc[trace_index, "p_picks (sec)"] = p_pick
    traces.loc[trace_index, "s_picks (sec)"] = s_pick
    traces.loc[trace_index, "pga"] = pga
    traces.loc[trace_index, "pga_time"] = pga_times
    traces.loc[trace_index, "pgv"] = pgv
    traces.loc[trace_index, "pgv_time"] = pgv_times

# others error: not no file or picking problem. result: txt file format problem
others_err_filter = error_file_df["reason"].str.contains(r"^\[Errno 2\]")
others_err_df = error_file_df[(~others_err_filter) & (~pick_again_filter)]

# drop pga NaN value
traces.dropna(inplace=True)
# traces.to_csv(f"{Afile_path}/2012-2020 traces with picking and label.csv",index=False)

# drop start_time not correct
for i in traces.index:
    try:
        pd.to_datetime(traces["start_time"][i], format="%Y%m%d%H%M%S")
    except:
        print(i, traces["start_time"][i])
        traces.drop([i], inplace=True)
# drop traces corresponds to wrong event:
for i in traces.index:
    trace_start_time = (
        int(str(traces["start_time"][i])[-6:-4]) * 60 * 60
        + int(str(traces["start_time"][i])[-4:-2]) * 60
        + int(str(traces["start_time"][i])[-2:])
    )
    event_time = (
        traces["hour"][i] * 60 * 60 + traces["minute"][i] * 60 + traces["second"][i]
    )
    if abs(trace_start_time - event_time) > 5 * 60:  # threshold 5 mins
        traces.drop([i], inplace=True)
traces.to_csv(
    f"{Afile_path}/1991-2020 traces with picking and label_filtered.csv", index=False
)


# traces station location doesn't exist
sta_path = "data/station information"
Afile_path = "data/Afile"
traces = pd.read_csv(
    f"{Afile_path}/1991-2020 traces with picking and label_filtered.csv"
)
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
sta_filter = traces["station_name"].isin(station_info["location_code"])
tmp_traces = traces[sta_filter]

# drop traces don't belongs to right event
event_duration_sec = 60
tmp_traces = pd.read_csv(
    f"{Afile_path}/1991-2020 traces with picking and label_new (sta location exist).csv"
)
tmp_traces.loc[tmp_traces.index, "p_picks (sec)"] = pd.to_timedelta(
    tmp_traces["p_picks (sec)"], unit="sec"
)
tmp_traces.loc[tmp_traces.index, "start_time"] = pd.to_datetime(
    tmp_traces["start_time"], format="%Y%m%d%H%M%S"
)
tmp_traces.loc[:, "record_time"] = (
    tmp_traces["start_time"] + tmp_traces["p_picks (sec)"]
)
correct_traces_filter = tmp_traces["record_time"] > pd.to_datetime(
    tmp_traces[["year", "month", "day", "hour", "minute", "second"]]
)
tmp_traces = tmp_traces[correct_traces_filter]

Traces = []
for eq_id in pd.unique(tmp_traces["EQ_ID"]):
    single_event_traces = tmp_traces[tmp_traces["EQ_ID"] == eq_id]
    sorted_indices = single_event_traces["epdis (km)"].sort_values().index
    single_event_traces = single_event_traces.loc[sorted_indices, :].reset_index(
        drop=True
    )
    # print(tmp_traces)
    end_time_filter1 = (
        single_event_traces["record_time"] >= single_event_traces["record_time"][0]
    )
    end_time_filter2 = single_event_traces["record_time"] <= single_event_traces[
        "record_time"
    ][0] + pd.to_timedelta(event_duration_sec, unit="s")
    single_event_traces = single_event_traces[end_time_filter1 & end_time_filter2]
    Traces.append(single_event_traces)
Traces_df = pd.concat(Traces)
Traces_df.to_csv(
    f"{Afile_path}/1991-2020 traces with picking and label_new (sta location exist)_filtered.csv",
    index=False,
)

# drop event don't have at least one trace
catalog = pd.read_csv(f"{Afile_path}/1991-2020 catalog.csv")
tmp_traces = pd.read_csv(
    f"{Afile_path}/1991-2020 traces with picking and label_new (sta location exist)_filtered.csv"
)

check_filter = catalog["EQ_ID"].isin(tmp_traces["EQ_ID"])

catalog.drop(catalog[~check_filter].index, inplace=True)
catalog.to_csv(f"{Afile_path}/final catalog (station exist)_filtered.csv", index=False)
