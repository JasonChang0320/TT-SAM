import numpy as np
import pandas as pd

from read_tsmip import *

sta_path = "../data/station_information"
station_info = pd.read_csv(f"{sta_path}/TSMIPstations.csv")
station_code = station_info["station"].str.extract(r"(.*?)[(]")
location_code = station_info["station"].str.extract(r"[(](.*?)[)]")
station_info.insert(1, "station_code", station_code.values)
station_info.insert(2, "location_code", location_code.values)
station_info.drop(["station"], axis=1, inplace=True)

# merge data from JC
for sta in ["CHY", "HWA", "ILA", "KAU", "TAP", "TCU", "TTN"]:
    tmp_info = pd.read_csv(f"{sta_path}/{sta}.csv", encoding="unicode_escape")
    tmp_info.columns = [
        "location_code",
        "station_location",
        "county",
        "district",
        "net",
        "longitude",
        "latitude",
        "elevation",
        "stamp code",
        "address",
    ]
    sta_filter = tmp_info["location_code"].isin(station_info["location_code"])
    add_df = tmp_info[~sta_filter][
        ["location_code", "latitude", "longitude", "elevation"]
    ]
    add_df.rename(columns={"elevation": "elevation (m)"}, inplace=True)
    add_df.insert(0, "network", "TSMIP")
    add_df.insert(1, "station_code", np.nan)
    station_info = pd.concat([station_info, add_df])

# merge data fron MH
# data1
station_code1 = pd.read_csv(f"{sta_path}/station_code.csv")
station_code2 = pd.read_csv(f"{sta_path}/tsmip_factor.csv")
merged_station_code = pd.merge(
    station_code1, station_code2, left_on="Station_Code", right_on="station_code"
)
sta_filter = merged_station_code["TSMIP_code"].isin(station_info["location_code"])
add_df = merged_station_code[~sta_filter][
    ["TSMIP_code", "Ins_longitude", "Ins_latitude", "Ins_elevation", "TSMIP_short_code"]
]
add_df.columns = [
    "location_code",
    "longitude",
    "latitude",
    "elevation (m)",
    "station_code",
]

save_index = []
for sta_code in add_df["location_code"].unique():
    save_index.append(
        add_df[add_df["location_code"] == sta_code]["location_code"].index[-1]
    )
uniqued_add_df = add_df.loc[save_index]
uniqued_add_df.insert(0, "network", "TSMIP")
station_info = pd.concat([station_info, uniqued_add_df])

# data2
CWBstation = pd.read_csv(f"{sta_path}/CWBstation.log", sep="\s+", header=None)
CWBstation.columns = [
    "location_code",
    "longitude",
    "latitude",
    "elevation (m)",
    "starttime",
    "endtime",
]
sta_filter = CWBstation["location_code"].isin(station_info["location_code"])
add_df = CWBstation[~sta_filter][
    ["location_code", "longitude", "latitude", "elevation (m)"]
]
add_df.insert(0, "network", "TSMIP")
add_df.insert(1, "station_code", np.nan)
station_info = pd.concat([station_info, add_df])
station_info.sort_values(by=["location_code"], inplace=True)
# station_info.to_csv(f"{sta_path}/TSMIPstations_new.csv", index=False)
