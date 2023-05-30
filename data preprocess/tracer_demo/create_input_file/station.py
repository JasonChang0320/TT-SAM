import pandas as pd


sta_path = "D:/TEAM_TSMIP/data/station information"
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
traces = pd.read_csv("../../events_traces_catalog/2009_2019_ok_picked_traces.csv")


station_name = pd.DataFrame(traces["station_name"].unique(), columns=["station_name"])

stations = pd.merge(
    station_name,
    station_info[["location_code", "latitude", "longitude", "elevation (m)"]],
    left_on="station_name",
    right_on="location_code",
    how="left",
)
stations.rename(columns={"elevation (m)": "elevation_m"}, inplace=True)

# KNM003 station location out of velocity model range
drop_station_name = "KNM003"
stations = stations[(stations["station_name"] != drop_station_name)].reset_index()


file_path = "station_input.sta"

with open(file_path, "w") as file:
    for i in range(len(stations)):
        file.write(
            f"{round(stations.longitude[i],3)} {round(stations.latitude[i],4)} {round(stations.elevation_m[i],3)} {stations.station_name[i]}\n"
        )
