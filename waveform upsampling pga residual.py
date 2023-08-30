import os

import matplotlib.pyplot as plt
import numpy as np

from read_tsmip import read_tsmip

resample_rate=200.0
path="D:/TEAM_TSMIP/data/waveform/1994/05"
files=os.listdir(path)
waveforms=[]
for file in files:
    if file.endswith(".txt"):
        waveforms.append(file)

origin_PGA=[]
resample_PGA=[]
Stream=[]
resample_Stream=[]
for waveform in waveforms:
    stream=read_tsmip(f"{path}/{waveform}")
    origin_pga=np.sqrt(stream.max()[0]**2+stream.max()[1]**2+stream.max()[2]**2)
    Stream.append(stream)
    origin_PGA.append(origin_pga)

    resample_stream=stream.resample(resample_rate,window="hann")
    resample_pga=np.sqrt(resample_stream.max()[0]**2+resample_stream.max()[1]**2+resample_stream.max()[2]**2)
    resample_Stream.append(resample_stream)
    resample_PGA.append(resample_pga)

origin_PGA=np.array(origin_PGA)
resample_PGA=np.array(resample_PGA)
fig,ax=plt.subplots()
ax.scatter(np.arange(len(origin_PGA)),origin_PGA,s=7,alpha=0.5,label="before")
ax.scatter(np.arange(len(resample_PGA)),resample_PGA,s=7,alpha=0.5,label="after")
ax.set_title("1994 May traces resample")
ax.set_ylabel("PGA (gal)")
ax.set_xlabel("trace index")
ax.legend()
#residuaL
fig,ax=plt.subplots()
ax.scatter(np.arange(len(origin_PGA)),origin_PGA-resample_PGA,s=10)
ax.set_ylabel("PGA (gal)")
ax.set_xlabel("trace index")
ax.set_title("1994 May traces resample residual (origin-resampled)")

