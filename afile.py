from read_tsmip import *
import re
import os

year=2012
month="01"
trace_path=f"data/waveform/{year}/{month}"
trace_folder=os.listdir(trace_path)

afile_name="201201A.DAT"
afile=f"data/Afile/{afile_name}"

# event_line_index=[]
event=[]
trace=[]
with open(afile) as f:
    for i, line in enumerate(f):
        if re.match(f"{year} .*",line):
            if i!=0:
                trace.append(tmp_trace)
            # event_line_index.append(i)
            event.append(line)
            tmp_trace=[]
        else:
            if line[46:59].strip() in trace_folder:
                tmp_trace.append(line)
    trace.append(tmp_trace) #remember to add the last event's traces


header_info = read_header(event[0])
trace_info = read_lines(trace[0])
