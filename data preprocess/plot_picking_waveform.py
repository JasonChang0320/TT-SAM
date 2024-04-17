import pandas as pd
from read_tsmip import read_tsmip
import matplotlib.pyplot as plt
from obspy.signal.trigger import ar_pick

ok_waveform_file = "1999_2019_final_traces_Vs30.csv"
year = 1999
traces = pd.read_csv(f"events_traces_catalog/{ok_waveform_file}")

waveform_path = "../data/waveform"
for i in traces.index:
    print(f"{i}/{len(traces)}")
    EQ_ID = str(traces["EQ_ID"][i])
    year = str(traces["year"][i])
    month = str(traces["month"][i])
    day = str(traces["day"][i])
    hour = str(traces["hour"][i])
    minute = str(traces["minute"][i])
    second = str(traces["second"][i])
    intensity = str(traces["intensity"][i])
    station_name = traces["station_name"][i]
    epdis = str(traces["epdis (km)"][i])
    file_name = traces["file_name"][i].strip()
    if len(month) < 2:
        month = "0" + month
    waveform = read_tsmip(f"{waveform_path}/{year}/{month}/{file_name}.txt")
    # p_pick=traces["p_pick_sec"][i]*waveform[0].stats.sampling_rate
    p_pick, _ = ar_pick(
        waveform[0],
        waveform[1],
        waveform[2],
        samp_rate=waveform[0].stats.sampling_rate,
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
    p_pick = p_pick * waveform[0].stats.sampling_rate
    fig, ax = plt.subplots(3, 1)
    for j in range(len(ax)):
        ax[j].plot(
            waveform[j].data[int(p_pick - 5 * 200) : int(p_pick + 30 * 200)], "k"
        )
        ax[j].axvline(x=5 * 200, color="r", linestyle="-")
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[1].set_ylabel(f"Amplitude (gal)")
    ax[2].set_xlabel(f"Time Sample (200Hz)")
    ax[0].set_title(f"EQ_ID:{EQ_ID},year:{year},month:{month},file_name:{file_name}")
    # fig.savefig(f"pick_result/EQ_ID_{EQ_ID}_{year}_{month}_{file_name}.png",dpi=300)
    plt.close()
