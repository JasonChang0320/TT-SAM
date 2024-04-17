import pandas as pd

start_year1=1999
end_year1=2008
traces1 = pd.read_csv(
    f"./events_traces_catalog/{start_year1}_{end_year1}_picked_traces_p_arrival_abstime_labeled_nostaoverlap.csv"
)
catalog1 = pd.read_csv(
    f"./events_traces_catalog/{start_year1}_{end_year1}_ok_events_p_arrival_abstime.csv"
)

start_year2=2009
end_year2=2019
traces2 = pd.read_csv(
    f"./events_traces_catalog/{start_year2}_{end_year2}_picked_traces_p_arrival_abstime_labeled_nostaoverlap.csv"
)
catalog2 = pd.read_csv(
    f"./events_traces_catalog/{start_year2}_{end_year2}_ok_events_p_arrival_abstime.csv"
)

final_trace=pd.concat([traces1,traces2])

final_catalog=pd.concat([catalog1,catalog2])

# final_trace.to_csv(f"./events_traces_catalog/{start_year1}_{end_year2}_final_traces.csv",index=False)
# final_catalog.to_csv(f"./events_traces_catalog/{start_year1}_{end_year2}_final_catalog.csv",index=False)
