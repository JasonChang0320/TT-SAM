import obspy
import pandas as pd


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

def read_header(header):
    if int(header[1:2]) == 9:
        header = header.replace("9", "199", 1)
    header_info = {
        "year": int(header[1:5]),
        "month": int(header[5:7]),
        "day": int(header[7:9]),
        "hour": int(header[9:11]),
        "minute": int(header[11:13]),
        "second": float(header[13:19]),
        "lat": float(header[19:21]),
        "lat_minute": float(header[21:26]),
        "lon": int(header[26:29]),
        "lon_minute": float(header[29:34]),
        "depth": float(header[34:40]),
        "magnitude": float(header[40:44]),
        "nsta": header[44:46].replace(" ", ""),
        "Pfilename": header[46:58].replace(" ", ""),
        "newNoPick": header[60:63].replace(" ", ""),
    }
    return header_info

def read_lines(lines):
    trace = []
    for line in lines:
        line = line.strip("\n")
        if len(line) < 109:  # missing ctime
            line = line + "   0.000"
        try:
            line_info = {
                "code": str(line[1:7]).replace(" ", ""),
                "epdis": float(line[7:13]),
                "az": int(line[13:17]),
                "phase": str(line[21:22]).replace(" ", ""),
                "ptime": float(line[23:30]),
                "pwt": int(line[30:32]),
                "stime": float(line[33:40]),
                "swt": int(line[40:42]),
                "lat": float(line[42:49]),
                "lon": float(line[49:57]),
                "gain": float(line[57:62]),
                "convm": str(line[62:63]).replace(" ", ""),
                "accf": str(line[63:75]).replace(" ", ""),
                "durt": float(line[75:79]),
                "cherr": int(line[80:83]),
                "timel": str(line[83:84]).replace(" ", ""),
                "rtcard": str(line[84:101]).replace(" ", ""),
                "ctime": str(line[101:109]).replace(" ", ""),
            }
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