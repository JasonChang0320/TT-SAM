import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

Afile_path = "./events_traces_catalog"

def plot_event_distribution(catalog, output_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.histplot(catalog, x="magnitude", hue="from", alpha=1, ax=ax)
    ax.set_title("Events Catalog", fontsize=20)
    ax.set_yscale("log")
    ax.set_xlabel("Magnitude", fontsize=13)
    ax.set_ylabel("Number of events", fontsize=13)
    if output_path:
        fig.savefig(f"{output_path}/event_distribution.png", dpi=300)
    return fig, ax

def plot_trace_distribution(trace, output_path=None):
    label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    pga_threshold = np.log10([0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10])
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.histplot(trace, x="pga", hue="from", alpha=1, ax=ax, bins=32)
    for i in range(len(pga_threshold) - 1):
        ax.text((pga_threshold[i] + pga_threshold[i + 1]) / 2, 10000, label[i])
    ax.vlines(pga_threshold[1:-1], 0, 40000, linestyles="dotted", color="k")
    ax.set_title("Traces catalog", fontsize=20)
    ax.set_yscale("log")
    ax.set_xlabel("PGA log(m/s^2)", fontsize=13)
    ax.set_ylabel("number of traces", fontsize=13)
    if output_path:
        fig.savefig(f"{output_path}/traces_distribution.png", dpi=300)
    return fig, ax

before_catalog = pd.read_csv(f"{Afile_path}/2009_2019_ok_events_p_arrival_abstime.csv")
after_catalog = pd.read_csv(f"{Afile_path}/1999_2019_final_catalog.csv")

before_catalog["from"] = "2009~2019 M>=3.5"
after_catalog["from"] = "1999~2008 M>=5.5"

catalog = pd.concat([before_catalog, after_catalog])
catalog.reset_index(inplace=True, drop=True)

fig, ax = plot_event_distribution(catalog, output_path=None)

###### trace

before_trace = pd.read_csv(
    f"{Afile_path}/2009_2019_picked_traces_p_arrival_abstime_labeled_nostaoverlap.csv"
)
after_trace = pd.read_csv(f"{Afile_path}/1999_2019_final_traces.csv")

before_trace["from"] = "2009~2019 M>=3.5"
after_trace["from"] = "1999~2008 M>=5.5"

trace = pd.concat([before_trace, after_trace])
trace.reset_index(inplace=True, drop=True)
label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
pga_threshold = np.log10([0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10])
fig, ax = plot_trace_distribution(trace, output_path=None)

print("high_intensity_rate")
print(
    "2009~2019:",
    len(before_trace.query(f"pga >={pga_threshold[2]}")) / len(before_trace),
)
print(
    "1999~2019:", len(after_trace.query(f"pga >={pga_threshold[2]}")) / len(after_trace)
)
