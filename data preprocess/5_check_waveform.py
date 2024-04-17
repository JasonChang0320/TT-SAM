import pandas as pd
import sys
import os

sys.path.append("..")
from read_tsmip import read_tsmip
from obspy.signal.trigger import ar_pick
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

start_year=1999
end_year=2008
start_index = 14031
Afile_path = "../data/Afile"
sta_path = "../data/station information"
waveform_path = "../data/waveform"
output_path="events_traces_catalog"
traces_file_name=f"{start_year}_{end_year}_target_traces.csv"
error_file_name=f"{start_year}_{end_year}_error_traces_file.csv"
traces = pd.read_csv(f"{output_path}/{traces_file_name}")
catalog = pd.read_csv(f"{output_path}/{start_year}_{end_year}_target_catalog.csv")


def ok_traces(traces=None, index=None):
    traces.loc[index, "quality_control"] = "y"
    win.destroy()


def broken_traces(traces=None, index=None):
    traces.loc[index, "quality_control"] = "n"
    win.destroy()
def quit(running):
    running.set(False)
    win.destroy()


if "quality_control" not in traces.columns:
    traces["quality_control"] = "TBD"
if os.path.isfile(f"{error_file_name}"):
  error_file=pd.read_csv(f"{error_file_name}")
else: 
  error_file = pd.DataFrame({'index':[]})
  error_file.to_csv(f"{error_file_name}", index=False)
for i in range(start_index,len(traces)):
    print(f"{i}/{len(traces)}")
    try:
        EQ_ID = str(traces["EQ_ID"][i])
        year = str(traces["year"][i])
        month = str(traces["month"][i])
        day = str(traces["day"][i])
        hour = str(traces["hour"][i])
        minute = str(traces["minute"][i])
        second = str(traces["second"][i])
        intensity = str(traces["intensity"][i])
        station_name= traces["station_name"][i]
        epdis=str(traces["epdis (km)"][i])
        file_name = traces["file_name"][i].strip()
        magnitude = catalog.query(f"EQ_ID=={EQ_ID}")["magnitude"].tolist()[0]
        if len(month) < 2:
            month = "0" + month
        waveform = read_tsmip(f"{waveform_path}/{year}/{month}/{file_name}.txt")
        # picking
        if i==8319: #1999~2008 index 8319 can't pick, kernel crushed
            continue
        p_pick,_ = ar_pick(
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
        if (p_pick-3)>0:
            start_time=int((p_pick-3)*waveform[0].stats.sampling_rate)
        else:
            start_time=0
        # plot
        fig, ax = plt.subplots(3, 1)
        fig.subplots_adjust(hspace=0.4)
        for j in range(len(ax)):
            # start_time=4000
            if (p_pick+30)*waveform[0].stats.sampling_rate<len(waveform[0].data):
                endtime=int((p_pick+30)*waveform[0].stats.sampling_rate)
                # endtime=4600
                ax[j].plot(waveform[j].times()[start_time:endtime], waveform[j].data[start_time:endtime], "k")
                ax[j].axvline(x=p_pick, color="r", linestyle="-")
            else:
                ax[j].plot(waveform[j].times()[start_time:endtime], waveform[j].data[start_time:endtime], "k")
                ax[j].axvline(x=p_pick, color="r", linestyle="-")
        ax[0].set_title(
            f"EQID:{EQ_ID}_{station_name}, {year} {month}/{day} {hour}:{minute}:{second}, magnitude: {magnitude}, intensity:{intensity}"
        )
        ax[1].set_title(f"epicentral distance: {epdis} km")
        ax[1].set_ylabel("gal")
        ax[-1].set_xlabel("time (sec)")
        plt.close()

        # GUI
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
        win.bind("<space>", lambda event: ok_traces(traces=traces, index=i))
        win.bind("<n>", lambda event: broken_traces(traces=traces, index=i))
        running = tk.BooleanVar(value=True)
        win.bind("<Key>", lambda event: quit(running) if event.keysym == "Escape" else None)
        win.mainloop()
        if running.get():
            pass
        else:
            print(f"stop at index:{i}")
            break
    except Exception as reason:
        print(file_name, f"year:{year},month:{month}, {reason}")
        row={"index":i,"year":int(year), "month":month, "file":file_name,"reason":reason}
        if i not in error_file["index"].values:
            error_file= pd.concat([error_file,pd.DataFrame(row, index=[0])],ignore_index=True)
        traces.loc[i, "quality_control"] = "n"
        continue

# traces.to_csv(f"{output_path}/{traces_file_name}", index=False)
# error_file.to_csv(f"{output_path}/{error_file_name}", index=False)
print("data saved")
