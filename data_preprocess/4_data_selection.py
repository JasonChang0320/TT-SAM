import pandas as pd

start_year = 1999
end_year = 2008
intensity_threshold = 4
magnitude_thrshold = 5.5


Afile_path = "../data/Afile"
sta_path = "../data/station_information"
traces = pd.read_csv(
    f"{Afile_path}/1991-2020 traces (no broken data, double event).csv"
)
catalog = pd.read_csv(f"{Afile_path}/1991-2020 catalog.csv")

# traces station location doesn't exist
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
sta_filter = traces["station_name"].isin(station_info["location_code"])
traces_exist_sta = traces[sta_filter]

# find Earthquake that at least 1 trace intensity > 4 & magnitude >=3.5
target_traces = traces_exist_sta.query(f"year>={start_year} & year<={end_year}")
EQ_ID = (
    target_traces.query(f"intensity >= {intensity_threshold}")["EQ_ID"]
    .unique()
    .tolist()
)
output_catalog = catalog.query(f"EQ_ID in {EQ_ID} & magnitude >= {magnitude_thrshold}")
output_traces = target_traces.copy()
EQ_ID = output_catalog["EQ_ID"].tolist()
output_traces = output_traces.query(f"EQ_ID in {EQ_ID}")

# check nan
output_traces.isnull().sum(axis=0)
output_catalog.isnull().sum(axis=0)
# plot magnitude hist & check intensity
output_catalog["magnitude"].hist(bins=16)
output_traces["intensity"].hist(bins=20)
output_traces["intensity"].value_counts()

# output_catalog.to_csv(f"events_traces_catalog/{start_year}_{end_year}_target_catalog.csv", index=False)
# output_traces.to_csv(f"events_traces_catalog/{start_year}_{end_year}_target_traces.csv", index=False)
