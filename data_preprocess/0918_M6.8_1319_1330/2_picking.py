import pandas as pd
import sys
import os
import numpy as np

sys.path.append("../..")
from obspy.signal.trigger import ar_pick
import matplotlib.pyplot as plt
import obspy
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re


start_index = 0


# ===================
def ok_traces(traces=None, index=None):
    traces.loc[index, "quality_control"] = "y"
    win.destroy()


def broken_traces(traces=None, index=None):
    traces.loc[index, "quality_control"] = "n"
    win.destroy()


def quit(running):
    running.set(False)
    win.destroy()


trace_catalog = pd.read_csv("traces_catalog.csv")

for k in range(start_index, len(trace_catalog["file_name"])):
    file_name = trace_catalog["file_name"][k]
    print(f"{k}/{len(trace_catalog)}")
    try:
        txt = f"../../data/0918_M6.8_1319_1330/ascii/{file_name}.asc"
        data = pd.read_csv(txt, sep="\s+", skiprows=1, header=None).to_numpy()

        with open(txt, "r") as f:
            picks = f.readlines()[0]
            picks = re.findall(r"\d+\.\d+", picks)
            picks = [np.round(float(number), 2) for number in picks]

        waveform = obspy.core.stream.Stream()
        channel = ["HLZ", "HLN", "HLE"]
        for i, chan in enumerate(channel):
            start = np.where(data == picks[2])[0][0]
            end = np.where(data == picks[3])[0][0]
            trace = obspy.core.trace.Trace(data[start:end, i + 1])

            trace.stats.network = "TW"
            # trace.stats.station = header[0][14:20]
            trace.stats.channel = chan

            trace.stats.sampling_rate = int(1 / abs(data[0, 0] - data[1, 0]))

            waveform.append(trace)

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
        trace_catalog.loc[k, "p_pick_sec"] = p_pick
        if (p_pick - 3) > 0:
            start_time = int((p_pick - 3) * waveform[0].stats.sampling_rate)
        else:
            start_time = 0
        # plot
        fig, ax = plt.subplots(3, 1)
        fig.subplots_adjust(hspace=0.4)
        for j in range(len(ax)):
            # start_time=4000
            if (p_pick + 30) * waveform[0].stats.sampling_rate < len(waveform[0].data):
                endtime = int((p_pick + 30) * waveform[0].stats.sampling_rate)
                # endtime=4600
                ax[j].plot(
                    waveform[j].times()[start_time:], waveform[j].data[start_time:], "k"
                )
                ax[j].axvline(x=p_pick, color="r", linestyle="-")
            else:
                ax[j].plot(
                    waveform[j].times()[start_time:], waveform[j].data[start_time:], "k"
                )
                ax[j].axvline(x=p_pick, color="r", linestyle="-")
        ax[0].set_title(f"{file_name}")
        ax[1].set_ylabel("gal")
        ax[-1].set_xlabel("time (sec)")
        plt.close()

        win = tk.Tk()
        win.attributes("-topmost", True)
        win.after(1, lambda: win.focus_force())
        win.title("check waveform")
        win.geometry("700x650+10+10")
        win.maxsize(1000, 700)
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        label = tk.Label(win, text="Press ESC to quit")
        label.pack()
        win.bind("<space>", lambda event: ok_traces(traces=trace_catalog, index=k))
        win.bind("<n>", lambda event: broken_traces(traces=trace_catalog, index=k))
        running = tk.BooleanVar(value=True)
        win.bind(
            "<Key>", lambda event: quit(running) if event.keysym == "Escape" else None
        )
        win.mainloop()
        if running.get():
            pass
        else:
            print(f"stop at index:{k}")
            break
    except Exception as reason:
        print(file_name, f"{reason}")
        row = {"index": i, "file": file_name, "reason": reason}
        if i not in error_file["index"].values:
            error_file = pd.concat(
                [error_file, pd.DataFrame(row, index=[0])], ignore_index=True
            )
        trace_catalog.loc[i, "quality_control"] = "n"
        continue
trace_catalog.to_csv(f"traces_catalog.csv", index=False)

# ========shift p_picking by velocity model to correct absolute time======
traces = pd.read_csv("traces_catalog.csv")
catalog = pd.read_csv("event_catalog.csv")

EQ_ID = os.listdir(f"../tracer_demo/2023_output")

traces.insert(0, "EQ_ID", 30792)

traces=pd.merge(
    catalog[["EQ_ID", "year", "month", "day", "hour", "minute", "second"]],
    traces,
    how="right",
    on="EQ_ID",
)
traces["p_arrival_abs_time"] = pd.to_datetime(
    traces[["year", "month", "day", "hour", "minute", "second"]]
)

colnames = [
    "evt_lon",
    "evt_lat",
    "evt_depth",
    "sta_lon",
    "sta_lat",
    "sta_elev",
    "p_arrival",
    "s_arrival",
]
for eq in EQ_ID:
    event_file_path = f"../tracer_demo/2023_output/{eq}/output.table"
    tracer_output = pd.read_csv(
        event_file_path, sep=r"\s+", names=colnames, header=None
    )
    trace_index = traces[traces["EQ_ID"] == int(eq)].index
    p_arrival = pd.to_timedelta(tracer_output["p_arrival"], unit="s")
    p_arrival.index = trace_index
    traces.loc[trace_index, "p_arrival_abs_time"] = (
        traces.loc[trace_index, "p_arrival_abs_time"] + p_arrival
    )
traces.to_csv(f"traces_catalog.csv", index=False)