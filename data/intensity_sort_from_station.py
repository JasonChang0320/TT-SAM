import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "../data_preprocess"
data = pd.read_csv(f"{path}/events_traces_catalog/1999_2019_final_traces_Vs30.csv")

stations = data["station_name"].unique()

for station in stations:
    print(station)
    tmp_data = data.query(f"station_name=='{station}'")
    fig, ax = plt.subplots()
    ax.hist(
        tmp_data["pga"],
        bins=30,
        ec="black",
    )
    hist, bins = np.histogram(tmp_data["pga"], bins=30)
    pga_threshold = np.log10(
        [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10]
    )
    label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    ax.vlines(pga_threshold[1:-1], 0, hist.max()+5, linestyles="dotted", color="k")
    for i in range(len(label)):
        if label[i] == "0":
            continue
        ax.text(
            ((pga_threshold[i] + pga_threshold[i + 1]) / 2) - 0.05, hist.max()+5, label[i]
        )
    ax.set_xlabel(r"PGA log(${m/s^2}$)", fontsize=12)
    ax.set_ylabel("Number of traces", fontsize=12)
    ax.set_title(f"station name: {station}", fontsize=15)
    # fig.savefig(f"{path}/each_station_distribution/{station}.png", dpi=300)
    plt.close()
