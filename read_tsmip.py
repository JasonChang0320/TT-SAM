import re

import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from obspy.signal.trigger import ar_pick


def read_tsmip(txt):
    # data = pd.read_fwf(txt, delim_whitespace=True, skiprows=10).to_numpy()
    data = pd.read_csv(txt, sep="\s+", skiprows=11, header=None).to_numpy()

    with open(txt, "r") as f:
        header = f.readlines()[:11]

    stream = obspy.core.stream.Stream()

    channel = ["HLZ", "HLN", "HLE"]
    for i, chan in enumerate(channel):
        trace = obspy.core.trace.Trace(data[:, i + 1])

        trace.stats.network = "TW"
        trace.stats.station = header[0][14:20]
        trace.stats.location = "10"
        trace.stats.channel = chan

        trace.stats.starttime = obspy.UTCDateTime(header[2][12:-1])
        trace.stats.sampling_rate = int(header[4][17:20])

        stream.append(trace)

    return stream


def classify_event_trace(afile_path, afile_name, trace_folder):
    events = []
    traces = []
    with open(afile_path) as f:
        for i, line in enumerate(f):
            if re.match(f"{afile_name[0:4]}.*", line):
                if i != 0:
                    traces.append(tmp_trace)
                # event_line_index.append(i)
                events.append(line)
                tmp_trace = []
            else:
                if line[46:59].strip() + ".txt" in trace_folder:
                    tmp_trace.append(line)
        traces.append(tmp_trace)  # remember to add the last event's traces
    return events, traces


def read_header(header, EQ_ID=None):
    # if int(header[1:2]) == 9:
    #     header = header.replace("9", "199", 1)
    # print(header)
    header_info = {
        "year": int(header[0:4]),
        "month": int(header[4:6]),
        "day": int(header[6:8]),
        "hour": int(header[8:10]),
        "minute": int(header[10:12]),
        "second": float(header[13:18]),
        "lat": float(header[18:20]),
        "lat_minute": float(header[20:25]),
        "lon": int(header[25:28]),
        "lon_minute": float(header[28:33]),
        "depth": float(header[33:39]),
        "magnitude": float(header[39:43]),
        "nsta": header[43:45].replace(" ", ""),
        "nearest_sta_dist (km)": header[45:50].replace(" ", "")
        # "Pfilename": header[46:58].replace(" ", ""),
        # "newNoPick": header[60:63].replace(" ", ""),
    }
    if EQ_ID:
        EQID_dict = {"EQ_ID": EQ_ID}
        EQID_dict.update(header_info)
        return EQID_dict
    return header_info


def read_lines(lines, EQ_ID=None):
    trace = []
    for line in lines:
        line = line.strip("\n")
        if len(line) < 109:  # missing ctime
            line = line + "   0.000"
        try:
            line_info = {
                "station_name": str(line[1:7]).replace(" ", ""),
                "intensity": str(line[8:9]),
                "epdis (km)": float(line[11:17]),
                "pga_z": float(line[18:25]),
                "pga_ns": float(line[25:32]),
                "pga_ew": float(line[32:39]),
                "record_time (sec)": float(line[39:45]),
                "file_name": str(line[46:58]),
                "instrument_code": str(line[58:63]),
                "start_time": int(line[64:78]),
                "sta_angle": int(line[81:84]),
            }
            if EQ_ID:
                EQID_dict = {"EQ_ID": EQ_ID}
                EQID_dict.update(line_info)
                line_info = EQID_dict
        except ValueError:
            print(line)
            continue
        trace.append(line_info)

    return trace


def read_afile(afile):
    with open(afile) as f:
        header = f.readline()
        lines = f.readlines()
    header_info = read_header(header)
    trace_info = read_lines(lines)
    event = obspy.core.event.Event()
    event.event_descriptions.append(obspy.core.event.EventDescription())
    origin = obspy.core.event.Origin(
        time=obspy.UTCDateTime(
            header_info["year"],
            header_info["month"],
            header_info["day"],
            header_info["hour"],
            header_info["minute"],
            header_info["second"],
        ),
        latitude=header_info["lat"] + header_info["lat_minute"] / 60,
        longitude=header_info["lon"] + header_info["lon_minute"] / 60,
        depth=header_info["depth"],
    )
    origin.header = header_info
    event.origins.append(origin)

    for trace in trace_info:
        try:
            rtcard = obspy.core.UTCDateTime(trace["rtcard"])
        except Exception as err:
            print(err)
            continue

        waveform_id = obspy.core.event.WaveformStreamID(station_code=trace["code"])
        for phase in ["P", "S"]:
            if float(trace[f"{phase.lower()}time"]) == 0:
                continue

            pick = obspy.core.event.origin.Pick(
                waveform_id=waveform_id,
                phase_hint=phase,
                time=rtcard + trace[f"{phase.lower()}time"],
            )
            pick.header = trace
            event.picks.append(pick)

    event.magnitudes = header_info["magnitude"]
    return event


def trace_pick_plot(
    trace, file_name, eq_num=None, output_path=None
):  # trace load by "read_tsmip"
    sampling_rate = trace[0].stats.sampling_rate
    error_file = {"year": [], "month": [], "file": [], "reason": []}
    year = trace[0].stats.starttime.year
    month = trace[0].stats.starttime.month
    file_name = file_name
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
    except Exception as reason:
        print(file_name, f"year:{year},month:{month}, {reason}")
        error_file["year"].append(year)
        error_file["month"].append(month)
        error_file["file"].append(file_name)
        error_file["reason"].append(reason)
        return error_file
    fig, ax = plt.subplots(3, 1)

    ax[0].set_title(
        f"station: {trace[0].stats.station}, start time: {trace[0].stats.starttime}"
    )

    for component in range(len(trace)):
        ax[component].plot(trace[component], "k")
        ymin, ymax = ax[component].get_ylim()
        ax[component].vlines(p_pick * sampling_rate, ymin, ymax, "r", label="P pick")
        ax[component].vlines(s_pick * sampling_rate, ymin, ymax, "g", label="S pick")
    ax[1].set_ylabel("acc. (gal)")
    ax[2].set_xlabel(f"time unit ({sampling_rate} Hz)")
    if eq_num:
        ax[1].set_title(f"eqrthquake_number: {eq_num}")
        fig.tight_layout()
    ax[0].legend()
    if output_path:
        fig.savefig(f"{output_path}/{file_name}.png")
        plt.close()
        return
    return p_pick, s_pick, fig


def get_peak_value(stream, pick_point=0,thresholds=None):
    data = [tr.data for tr in stream]
    data = np.array(data)
    data=data[:,pick_point:]
    vector = np.linalg.norm(data, axis=0)

    peak = max(vector)
    peak_time = np.argmax(vector, axis=0)
    peak_time+=pick_point
    peak = np.log10(peak / 100)

    exceed_times = np.zeros(5)
    if thresholds is not None:
        for i, threshold in enumerate(thresholds):
            try:
                exceed_times[i] = next(
                    x for x, val in enumerate(vector) if val > threshold
                )
            except Exception as err:
                print(err)

    return peak, peak_time


def get_integrated_stream(stream):
    stream_intergrated = stream.copy()
    stream_intergrated.detrend(type="demean")
    stream_intergrated.filter("lowpass", freq=10)
    stream_intergrated.taper(max_percentage=0.05, type="cosine")
    stream_intergrated.integrate()
    stream_intergrated.filter("bandpass", freqmin=0.075, freqmax=10)
    return stream_intergrated


def cut_traces(
    traces,
    eq_id,
    waveform_path,
    before_p_sec=5,
    trace_length_sec=30,
    target_sampling_rate=200,
    waveform_type="acc",
):  # traces is dataframe
    traces_info = {
        "traces": [],
        "p_picks": [],
        "start_time": [],
        "pga": [],
        "pgv": [],
        "pga_time": [],
        "pgv_time": [],
    }
    traces_filter = traces["EQ_ID"] == eq_id
    tmp_traces = traces[traces_filter]

    sorted_indices = tmp_traces["p_arrival_abs_time"].sort_values().index
    tmp_traces = tmp_traces.loc[sorted_indices, :].reset_index(drop=True)

    year = tmp_traces["year"][0]
    month = tmp_traces["month"][0]
    file_name = tmp_traces["file_name"][0]

    if len(str(month)) < 2:
        month = "0" + str(month)
    path = f"{waveform_path}/{year}/{month}"
    file_name = file_name.strip()
    stream = read_tsmip(f"{path}/{file_name}.txt")
    sampling_rate = stream[0].stats["sampling_rate"]
    stream.detrend(type="demean")  # baseline correction
    stream.filter("lowpass", freq=10)  # filter
    if sampling_rate != target_sampling_rate:
        stream.resample(target_sampling_rate, window="hann")
    if waveform_type == "acc":
        pass
    elif waveform_type == "vel":
        stream = get_integrated_stream(stream)
    elif waveform_type == "dis":
        stream = get_integrated_stream(stream)
        stream = get_integrated_stream(stream)

    trace = np.transpose(np.array(stream)) / 100  # cm/s^2 to m/s^2

    trace_length_point = int(trace_length_sec * target_sampling_rate)
    first_start_cut_point = int(
    np.round(
        (
            tmp_traces["p_pick_sec"][0] - pd.Timedelta(seconds=before_p_sec)
        ).total_seconds(),
        2,
    )
    * target_sampling_rate
    )
    abs_cut_starttime = tmp_traces["p_arrival_abs_time"][0] - pd.Timedelta(
    seconds=before_p_sec
    )
    if first_start_cut_point < 0:
        first_start_cut_point = 0
        abs_cut_starttime = (
            tmp_traces["p_arrival_abs_time"][0] - tmp_traces["p_pick_sec"][0]
        )

    if first_start_cut_point + trace_length_point > len(trace):  # zero padding
        init_trace = trace[first_start_cut_point:, :]
        init_trace = np.pad(
            init_trace, ((0, trace_length_point - len(init_trace)), (0, 0)), "constant"
        )
    else:
        init_trace = trace[
            first_start_cut_point : first_start_cut_point + trace_length_point, :
        ]

    p_picks_point = int(
        np.round(
            tmp_traces["p_pick_sec"][0].total_seconds() * target_sampling_rate, 0
        )
        - first_start_cut_point
    )
    pga_time = int(tmp_traces["pga_time"][0] - first_start_cut_point)
    pgv_time = int(tmp_traces["pgv_time"][0] - first_start_cut_point)

    traces_info["traces"].append(init_trace)
    traces_info["p_picks"].append(p_picks_point)
    traces_info["pga"].append(tmp_traces["pga"][0])
    traces_info["pgv"].append(tmp_traces["pgv"][0])
    traces_info["pga_time"].append(pga_time)
    traces_info["pgv_time"].append(pgv_time)
    traces_info["start_time"].append(str(abs_cut_starttime))

    if len(tmp_traces) > 1:
        # print("more than  1 traces")
        for i in range(1, len(tmp_traces)):
            year = tmp_traces["year"][i]
            month = tmp_traces["month"][i]
            file_name = tmp_traces["file_name"][i]

            if len(str(month)) < 2:
                month = "0" + str(month)
            path = f"{waveform_path}/{year}/{month}"
            file_name = file_name.strip()
            stream = read_tsmip(f"{path}/{file_name}.txt")
            sampling_rate = stream[0].stats["sampling_rate"]
            if sampling_rate != target_sampling_rate:
                stream.resample(target_sampling_rate, window="hann")
            stream.detrend(type="demean")  # baseline correction
            stream.filter("lowpass", freq=10)  # filter

            if waveform_type == "acc":
                pass
            elif waveform_type == "vel":
                stream = get_integrated_stream(stream)
            elif waveform_type == "dis":
                stream = get_integrated_stream(stream)
                stream = get_integrated_stream(stream)

            trace = np.transpose(np.array(stream)) / 100  # cm/s^2 to m/s^2
            window_cut_time = (
                        tmp_traces["p_pick_sec"][i] 
                        - (tmp_traces["p_arrival_abs_time"][i] - abs_cut_starttime)
                    ).total_seconds()
            start_cut_point = int(window_cut_time * target_sampling_rate)
            if window_cut_time < 0:
                # print("pad at the beginning")
                end_cut_time = trace_length_point + start_cut_point
                if end_cut_time <= 0:
                    cutting_trace = np.zeros([trace_length_point, 3])
                else:
                    cutting_trace = trace[:end_cut_time, :]
                    cutting_trace = np.pad(
                        cutting_trace, ((abs(start_cut_point), 0), (0, 0)), "constant"
                    )
            else:
                # print("no pad at the beginning")
                cutting_trace = trace[
                    start_cut_point : start_cut_point + trace_length_point, :
                ]
            if len(cutting_trace) < trace_length_point:  # waveform too short, padding
                cutting_trace = np.pad(
                    cutting_trace,
                    ((0, trace_length_point - len(cutting_trace)), (0, 0)),
                    "constant",
                )
            p_picks_point = int(
                np.round(
                    tmp_traces["p_pick_sec"][i].total_seconds()
                    * target_sampling_rate,
                    0,
                )
                - start_cut_point
            )
            pga_time = int(tmp_traces["pga_time"][i] - start_cut_point)
            pgv_time = int(tmp_traces["pgv_time"][i] - start_cut_point)

            traces_info["traces"].append(cutting_trace)
            traces_info["p_picks"].append(p_picks_point)
            traces_info["pga"].append(tmp_traces["pga"][i])
            traces_info["pgv"].append(tmp_traces["pgv"][i])
            traces_info["pga_time"].append(pga_time)
            traces_info["pgv_time"].append(pgv_time)
            traces_info["start_time"].append(str(abs_cut_starttime))
    return tmp_traces, traces_info


def plot_cutting_event(
    traces_catalog,
    traces_info,
    subplot_max_num=25,
    trace_length_sec=30,
    sampling_rate=200,
    close_fig=False,
):
    eq_id = traces_catalog["EQ_ID"][0]
    trace_length_point = trace_length_sec * sampling_rate
    if len(traces_catalog) > 1:
        if len(traces_catalog) > subplot_max_num:
            fig, ax = plt.subplots(subplot_max_num, 1, figsize=(14, 7))
        else:
            fig, ax = plt.subplots(len(traces_catalog), 1, figsize=(14, 7))
        ax[0].plot(traces_info["traces"][0])
        ymin, ymax = ax[0].get_ylim()
        ax[0].vlines(traces_info["p_picks"][0], ymin, ymax, "r")
        ax[0].set_title(f"EQ ID: {eq_id}", fontsize=20)
        ax[0].set_yticks([])

    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(traces_info["traces"][0])
        ymin, ymax = ax.get_ylim()
        ax.vlines(traces_info["p_picks"][0], ymin, ymax, "r")
        ax.set_title(f"EQ ID: {eq_id}", fontsize=20)
    for i in range(1, len(traces_catalog)):
        if i >= subplot_max_num:
            continue
        ax[i].plot(traces_info["traces"][i])
        ax[i].set_yticks([])
        ymin, ymax = ax[i].get_ylim()
        if traces_info["p_picks"][i] < trace_length_point:
            ax[i].vlines(traces_info["p_picks"][i], ymin, ymax, "r")
    if close_fig:
        plt.close()
    return fig
