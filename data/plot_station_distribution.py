import pandas as pd
from visualize import plot_station_distribution

data = pd.read_csv("../data_preprocess/events_traces_catalog/1999_2019_final_traces_Vs30.csv")

unique_station = data.drop_duplicates(subset="station_name")

fig,ax=plot_station_distribution(stations=unique_station,title="TSMIP station distribution")

# fig.savefig(f"paper image/TSMIP_station_distribution.png",dpi=300)
