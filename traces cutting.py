from read_tsmip import *
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import timedelta

Afile_path="data/Afile"
catalog=pd.read_csv(f"{Afile_path}/final catalog.csv")
traces=pd.read_csv(f"{Afile_path}/2012-2020 traces with picking and label.csv")
traces.loc[traces.index,"p_picks (sec)"]=pd.to_timedelta(traces["p_picks (sec)"],unit="sec")
traces.loc[traces.index,"start_time"]=pd.to_datetime(traces['start_time'], format='%Y%m%d%H%M%S')

eq_id=2561
eq_id=7613
eq_id=4682
eq_id=423
before_p_sec=5
trace_length_sec=30

traces_filter=(traces["EQ_ID"]==eq_id)
tmp_traces=traces[traces_filter]

# tmp_traces["p_picks (sec)"]=pd.to_timedelta(tmp_traces["p_picks (sec)"],unit="sec")
# tmp_traces['start_time'] = pd.to_datetime(tmp_traces['start_time'], format='%Y%m%d%H%M%S')
sorted_indices = (tmp_traces["start_time"] + tmp_traces["p_picks (sec)"])\
                    .sort_values().index
tmp_traces=tmp_traces.loc[sorted_indices, :].reset_index(drop=True)
year=tmp_traces["year"][0]
month=tmp_traces["month"][0]
path=f"data/waveform/{year}/{month}"
file_name=tmp_traces["file_name"][0]

if len(str(month))<2:
    month="0"+str(month)
path=f"data/waveform/{year}/{month}"
file_name=file_name.strip()
stream=read_tsmip(f"{path}/{file_name}.txt")
sampling_rate=stream[0].stats["sampling_rate"]
trace=np.transpose(np.array(stream))/100 #cm/s^2 to m/s^2

trace_length_point=int(trace_length_sec*sampling_rate)
first_start_cut_point=int((np.round(tmp_traces["p_picks (sec)"][0].total_seconds(),2)-before_p_sec)*sampling_rate)
if first_start_cut_point<0:
    first_start_cut_point=0
if first_start_cut_point+trace_length_point>len(trace): #zero padding
    init_trace=trace[first_start_cut_point:,:]
    init_trace=np.pad(init_trace,((0,trace_length_point-len(init_trace)),(0,0)),"constant")
else:
    init_trace=trace[first_start_cut_point:first_start_cut_point+trace_length_point,:]

p_picks_point=int(np.round(tmp_traces["p_picks (sec)"][0].total_seconds()*sampling_rate,0)-first_start_cut_point)
pga_time=int(tmp_traces["pga_time"][0]-first_start_cut_point)

if len(tmp_traces)>1:
    fig,ax=plt.subplots(len(tmp_traces),1,figsize=(14,7))
    ax[0].plot(init_trace)
    ymin,ymax=ax[0].get_ylim()
    ax[0].vlines(p_picks_point,ymin,ymax,"r")
    ax[0].set_title(f"EQ ID: {eq_id}",fontsize=20)
    ax[0].set_yticks([])

else:
    fig,ax=plt.subplots(figsize=(14,7))
    ax.plot(init_trace)
    ymin,ymax=ax.get_ylim()
    ax.vlines(p_picks_point,ymin,ymax,"r")
    ax.scatter(pga_time,tmp_traces["pga"][0])

if len(tmp_traces)>1:
    print("more than  1 traces")
    for i in range(1,len(tmp_traces)):

        year=tmp_traces["year"][i]
        month=tmp_traces["month"][i]
        path=f"data/waveform/{year}/{month}"
        file_name=tmp_traces["file_name"][i]

        if len(str(month))<2:
            month="0"+str(month)
        path=f"data/waveform/{year}/{month}"
        file_name=file_name.strip()
        stream=read_tsmip(f"{path}/{file_name}.txt")
        sampling_rate=stream[0].stats["sampling_rate"]
        trace=np.transpose(np.array(stream))/100 #cm/s^2 to m/s^2
        window_cut_time=(tmp_traces["start_time"][0]-tmp_traces["start_time"][i]).total_seconds()+\
                        (first_start_cut_point/sampling_rate)
        start_cut_point=int(window_cut_time*sampling_rate)
        if window_cut_time<0:
            print("pad at the beginning")
            end_cut_time=trace_length_point+start_cut_point
            if end_cut_time<=0:
                cutting_trace=np.zeros([trace_length_point,3])
            else:
                cutting_trace=trace[:end_cut_time,:]
                cutting_trace=np.pad(cutting_trace,((abs(start_cut_point),0),(0,0)),"constant")
        else:
            print("no pad")
            cutting_trace=trace[start_cut_point:start_cut_point+trace_length_point,:]
        p_picks_point=int(np.round(tmp_traces["p_picks (sec)"][i].total_seconds()*sampling_rate,0)-start_cut_point)
        pga_time=int(tmp_traces["pga_time"][i]-start_cut_point)

        print(p_picks_point)
        ax[i].plot(cutting_trace)
        ax[i].set_yticks([])
        ymin,ymax=ax[i].get_ylim()
        if p_picks_point<trace_length_point:
            ax[i].vlines(p_picks_point,ymin,ymax,"r")



# fig,ax=plt.subplots(2,1,figsize=(14,7))
# ax[0].plot(init_trace)
# ax[1].plot(cutting_trace)


