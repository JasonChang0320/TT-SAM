import pandas as pd

station_info = pd.read_csv("../../data/station_information/TSMIPstations_new.csv")
traces_info_with_vs30 = pd.read_csv(
    "../events_traces_catalog/1999_2019_final_traces_Vs30.csv"
)

pick_result = pd.read_csv("../../data/0918_M6.8_1319_1330/result.csv", header=None)
pick_result.drop(0, axis=1, inplace=True)
pick_result.columns = ["file_name", "pick_result"]

ok_traces = pick_result.query("pick_result=='y'")

ok_traces["station_code"] = ok_traces["file_name"].str[3:7]

ok_traces = pd.merge(
    ok_traces,
    station_info[["station_code", "location_code"]],
    on="station_code",
    how="left",
)


for i in ok_traces.index:
    if pd.isna(ok_traces["location_code"][i]):
        if ok_traces["station_code"][i][0] == "A":
            ok_traces["location_code"][i] = "TAP" + ok_traces["station_code"][i][1:]
        if ok_traces["station_code"][i][0] == "B":
            ok_traces["location_code"][i] = "TCU" + ok_traces["station_code"][i][1:]
        if ok_traces["station_code"][i][0] == "C":
            ok_traces["location_code"][i] = "CHY" + ok_traces["station_code"][i][1:]
        if ok_traces["station_code"][i][0] == "D":
            ok_traces["location_code"][i] = "KAU" + ok_traces["station_code"][i][1:]
        if ok_traces["station_code"][i][0] == "E":
            ok_traces["location_code"][i] = "ILA" + ok_traces["station_code"][i][1:]
        if ok_traces["station_code"][i][0] == "F":
            ok_traces["location_code"][i] = "HWA" + ok_traces["station_code"][i][1:]
        if ok_traces["station_code"][i][0] == "G":
            ok_traces["location_code"][i] = "TTN" + ok_traces["station_code"][i][1:]

ok_traces = pd.merge(
    ok_traces,
    station_info[["location_code", "latitude", "longitude", "elevation (m)"]],
    on="location_code",
    how="left",
)

ok_traces = pd.merge(
    ok_traces,
    traces_info_with_vs30[["station_name", "Vs30"]].drop_duplicates(
        subset="station_name"
    ),
    left_on="location_code",
    right_on="station_name",
    how="left",
)

ok_traces.dropna(inplace=True)
ok_traces.drop(["station_code","location_code","pick_result"],axis=1,inplace=True)

ok_traces.to_csv("../0918_M6.8_1319_1330/traces_catalog.csv",index=None)
