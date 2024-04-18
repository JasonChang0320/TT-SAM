import os

import pandas as pd

from read_tsmip import *

Afile_path = "../data/Afile"
wavefrom_path = "../data/waveform"
Events = []
Traces = []
for year in os.listdir(f"{wavefrom_path}"):
    for month in [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ]:
        trace_path = f"{wavefrom_path}/{year}/{month}"
        trace_folder = os.listdir(trace_path)

        afile_name = f"{year}{month}A.DAT"
        afile_path = f"{Afile_path}/{afile_name}"
        events, traces = classify_event_trace(afile_path, afile_name, trace_folder)
        Events.extend(events)
        Traces.extend(traces)

# Events
event_dict_inlist = []
for eq_id, event in enumerate(Events):
    header_info = read_header(event, EQ_ID=str(eq_id + 1))
    event_dict_inlist.append(header_info)

event_df = pd.DataFrame.from_dict(event_dict_inlist)
# event_df.to_csv("{Afile_path}/1991-2020 catalog.csv", index=False)

# Traces
for i in range(len(Traces)):
    if i == 0:
        trace_info = read_lines(Traces[i], EQ_ID=str(i + 1))
    else:
        tmp_trace_info = read_lines(Traces[i], EQ_ID=str(i + 1))
        trace_info.extend(tmp_trace_info)

trace_df = pd.DataFrame.from_dict(trace_info)
# check traces belong to TSMIP:
trace_df = trace_df[trace_df["station_name"].str.len() == 6]

# trace_df.to_csv("{Afile_path}/1991-2020 traces.csv", index=False)
