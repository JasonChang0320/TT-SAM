import pandas as pd
import os

start_year=1999
end_year=2008
event = pd.read_csv(f"../../events_traces_catalog/{start_year}_{end_year}_ok_events.csv")
sta_path = "D:/TEAM_TSMIP/data/station_information"
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
traces = pd.read_csv(f"../../events_traces_catalog/{start_year}_{end_year}_ok_picked_traces.csv")

event["longitude"] = event["lon"] + (event["lon_minute"] / 60)
event["latitude"] = event["lat"] + (event["lat_minute"] / 60)

for eq_id in event["EQ_ID"]:
    #
    if eq_id==29363: #event location out of velocity model range
        continue
    tmp_event = event[event["EQ_ID"] == eq_id].reset_index()
    tmp_trace = traces[traces["EQ_ID"] == eq_id]
    tmp_trace_sta_lon_lat = pd.merge(
        tmp_trace,
        station_info[["location_code", "latitude", "longitude", "elevation (m)"]],
        left_on="station_name",
        right_on="location_code",
        how="left",
    )
    #
    drop_station_name = "KNM003" #station_location velocity model range
    tmp_trace_sta_lon_lat = tmp_trace_sta_lon_lat[(tmp_trace_sta_lon_lat["station_name"] != drop_station_name)].reset_index()
    tmp_trace_sta_lon_lat.rename(columns={'elevation (m)': 'elevation_m'}, inplace=True)
    folder_path = f"{start_year}_{end_year}_input/{eq_id}/"

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    event_file_path = folder_path + "event_input.evt"
    with open(event_file_path, "w") as file:
        file.write(
            f"{round(tmp_event.longitude[0],3)}  {round(tmp_event.latitude[0],3)}  {tmp_event.depth[0]}\n"
        )

    station_file_path = folder_path + "station_input.sta"
    with open(station_file_path, "w") as file:
        for i in range(len(tmp_trace_sta_lon_lat)):
            file.write(
                f"{round(tmp_trace_sta_lon_lat.longitude[i],3)} {round(tmp_trace_sta_lon_lat.latitude[i],4)} {round(tmp_trace_sta_lon_lat.elevation_m[i],3)} {tmp_trace_sta_lon_lat.station_name[i]}\n"
            )

