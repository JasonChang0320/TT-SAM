import pandas as pd

start_year=1999
end_year=2008
traces = pd.read_csv(
    f"./events_traces_catalog/{start_year}_{end_year}_picked_traces_p_arrival_abstime_labeled.csv"
)
catalog = pd.read_csv(
    f"./events_traces_catalog/{start_year}_{end_year}_ok_events_p_arrival_abstime.csv"
)

traces["instrument_priority"] = traces["instrument_code"].map(
    {" SMTA": 1, " CVA ": 2, " NANO": 3, " A900": 4, " ETNA": 5, " K2  ": 6, " REFT": 7}
)
# 抓出eq_id & station_name 相同的 trace
overlap_trace = pd.DataFrame()
for eq_id in catalog["EQ_ID"]:
    tmp_traces = traces.query(f"EQ_ID == {eq_id}")
    counts = tmp_traces["station_name"].value_counts()

    target_station = counts[counts > 1].index.tolist()

    mask = tmp_traces["station_name"].isin(target_station)

    tmp_overlap_trace = tmp_traces[mask]

    overlap_trace = pd.concat([overlap_trace, tmp_overlap_trace])

# 將instrument 編號，設定優先順序
insrument_priority = traces["instrument_code"].value_counts().index.tolist()

overlap_trace_sorted = overlap_trace.sort_values("instrument_priority")
chosen_trace = overlap_trace_sorted.drop_duplicates(
    ["station_name", "EQ_ID"], keep="first"
)
chosen_trace = chosen_trace.sort_index()

# 找原始df和有重疊df的差集，最後加回有重疊但最後留下來的trace
differ_set = pd.concat([traces, overlap_trace]).drop_duplicates(
    ["station_name", "EQ_ID"], keep=False
)
final_trace = pd.concat([differ_set, chosen_trace]).sort_index()


# final_trace.to_csv(
#     f"./events_traces_catalog/{start_year}_{end_year}_picked_traces_p_arrival_abstime_labeled_nostaoverlap.csv",
#     index=False,
# )

