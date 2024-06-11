import pandas as pd
import json
import sys
sys.path.append("..")
from analysis import Triggered_Map

i = 1
# for mask_after_sec in range(1,11):
mask_after_sec = 10
with open(f"model_input/{mask_after_sec}_sec/{i}.json", "r") as json_file:
    data = json.load(json_file)

station = data["sta"]
station_info = pd.DataFrame(
    station, columns=["latitude", "longitude", "elevation", "Vs30"]
)
condition = (
    (station_info["latitude"] == 0)
    & (station_info["longitude"] == 0)
    & (station_info["elevation"] == 0)
    & (station_info["Vs30"] == 0)
)
station_info = station_info.drop(station_info[condition].index)

station_info["event_lon"]=121.67
station_info["event_lat"]=23.77
station_info['magnitude']=7.2

fig,ax_map=Triggered_Map.plot_station_map(trace_info=station_info,min_epdis=10.87177078,sec=mask_after_sec)

ax_map.set_title(f"After {mask_after_sec} seconds")

# fig.savefig(f"triggered_station/{mask_after_sec}_sec_triggered_station.png", dpi=300)
