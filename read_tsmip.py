import obspy
import pandas as pd
import re


def read_tsmip(txt):
    data = pd.read_fwf(txt, delim_whitespace=True, skiprows=11).to_numpy()

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

def classify_event_trace(afile_path,afile_name,trace_folder):
    events=[]
    traces=[]
    with open(afile_path) as f:
        for i, line in enumerate(f):
            if re.match(f"{afile_name[0:4]}.*",line):
                if i!=0:
                    traces.append(tmp_trace)
                # event_line_index.append(i)
                events.append(line)
                tmp_trace=[]
            else:
                if line[46:59].strip() in trace_folder:
                    tmp_trace.append(line)
        traces.append(tmp_trace) #remember to add the last event's traces
    return events,traces

def read_header(header,EQ_ID=None):
    if int(header[1:2]) == 9:
        header = header.replace("9", "199", 1)
    header_info = {
        "year": int(header[0:4]),
        "month": int(header[4:6]),
        "day": int(header[6:8]),
        "hour": int(header[8:10]),
        "minute": int(header[10:12]),
        "second": float(header[12:18]),
        "lat": float(header[18:20]),
        "lat_minute": float(header[20:25]),
        "lon": int(header[25:28]),
        "lon_minute": float(header[28:33]),
        "depth": float(header[33:39]),
        "magnitude": float(header[39:43]),
        "nsta": header[43:45].replace(" ", ""),
        "nearest_sta_dist (km)":header[45:50].replace(" ","")
        # "Pfilename": header[46:58].replace(" ", ""),
        # "newNoPick": header[60:63].replace(" ", ""),
    }
    if EQ_ID:
        EQID_dict={"EQ_ID":EQ_ID}
        EQID_dict.update(header_info)
        return EQID_dict
    return header_info

def read_lines(lines,EQ_ID=None):
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
                "sta_angle": int(line[81:84])
            }
            if EQ_ID:
                EQID_dict={"EQ_ID":EQ_ID}
                EQID_dict.update(line_info)
                line_info=EQID_dict
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