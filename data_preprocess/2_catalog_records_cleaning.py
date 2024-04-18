import matplotlib.pyplot as plt
import pandas as pd

Afile_path = "../data/Afile"
##############clean broken data and small data##############

traces_catalog = pd.read_csv(f"{Afile_path}/1991-2020 traces.csv")
acc_filter = (
    (traces_catalog["pga_z"] == 0)
    | (traces_catalog["pga_ns"] == 0)
    | (traces_catalog["pga_ew"] == 0)
    | (traces_catalog["pga_z"] < 2.5)
    | (traces_catalog["pga_ns"] < 2.5)
    | (traces_catalog["pga_ew"] < 2.5)
    | (traces_catalog["pga_z"] > 1300)
    | (traces_catalog["pga_ns"] > 1300)
    | (traces_catalog["pga_ew"] > 1300)
)
broken_traces = traces_catalog[acc_filter]
# broken_traces.to_csv(f"{Afile_path}/1991-2020 broken traces.csv", index=False)

traces_catalog.drop(traces_catalog[acc_filter].index, inplace=True)
# traces_catalog.to_csv(f"{Afile_path}/1991-2020 traces no broken data.csv", index=False)

##############find double event traces##############
catalog = pd.read_csv(f"{Afile_path}/1991-2020 catalog.csv")
traces_ljoin_catalog = pd.merge(
    catalog[["EQ_ID", "year", "month", "day", "hour", "minute", "second"]],
    traces_catalog,
    on="EQ_ID",
)

double_traces_catalog = pd.DataFrame()
for year in range(1991, 2021):
    for month in range(1, 13):
        time_filter = (traces_ljoin_catalog["year"] == year) & (
            traces_ljoin_catalog["month"] == month
        )
        tmp_catalog = traces_ljoin_catalog[time_filter]
        file_name_num = tmp_catalog["file_name"].value_counts()
        double_event = file_name_num[file_name_num > 1]
        same_filename_filter = tmp_catalog["file_name"].isin(double_event.index)
        double_traces = tmp_catalog[same_filename_filter]
        double_traces_catalog = pd.concat([double_traces_catalog, double_traces])
# double_traces_catalog.to_csv(f"{Afile_path}/1991-2020 double traces.csv", index=False)

# clean trace double event
traces_catalog = pd.read_csv(f"{Afile_path}/1991-2020 traces no broken data.csv")
catalog = pd.read_csv(f"{Afile_path}/1991-2020 catalog.csv")
traces_catalog_merge = pd.merge(
    catalog[["EQ_ID", "year", "month", "day", "hour", "minute", "second"]],
    traces_catalog,
    on="EQ_ID",
)

double_event = pd.read_csv(f"{Afile_path}/1991-2020 double traces.csv")

final_traces_catalog = pd.concat(
    [traces_catalog_merge, double_event, double_event]
).drop_duplicates(keep=False)
# final_traces_catalog.to_csv(
#     f"{Afile_path}/1991-2020 traces (no broken data, double event).csv", index=False
# )
