import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

trace = pd.read_csv("./events_traces_catalog/1999_2019_final_traces_Vs30.csv")
catalog = pd.read_csv("./events_traces_catalog/1999_2019_final_catalog.csv")

fig, ax = plt.subplots(figsize=(7, 7))
ax.hist(
    [trace.query("year>=2009")["pga"],trace.query("year<2009")["pga"]],
    bins=25,
    edgecolor="black",
    stacked=True,
    label=["origin","increased"],
)
ax.legend(loc='best')
ax.set_yscale("log")
label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
pga_threshold = np.log10(
    [0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0,10])
ax.vlines(pga_threshold[1:-1], 0, 35000, linestyles="dotted", color="k")
for i in range(len(pga_threshold) - 1):
    ax.text((pga_threshold[i] + pga_threshold[i + 1]) / 2, 15000, label[i])
ax.set_ylabel("number of trace")
ax.set_xlabel("log(PGA (m/s2))")
ax.set_title("TSMIP data PGA distribution")
# fig.savefig("./events_traces_catalog/pga distribution.png",dpi=300)

fig, ax = plt.subplots(figsize=(7, 7))
ax.hist(
    [catalog.query("year>=2009")["magnitude"],catalog.query("year<2009")["magnitude"]],
    bins=25,
    edgecolor="black",
    stacked=True,
    label=["origin","increased"],
)
ax.legend(loc='best')
ax.set_yscale("log")
ax.set_ylabel("number of event")
ax.set_xlabel("magnitude")
ax.set_title("TSMIP data magnitude distribution")
# fig.savefig("./events_traces_catalog/magnitude distribution.png",dpi=300)