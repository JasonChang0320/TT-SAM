import pandas as pd
import os

#before run this script, you need to go to "tracer_demo/":
#use "input_file.py" to create input data
# calculate p wave arrival by velocity model(Huang et al., 2014)
#paper link: https://www.sciencedirect.com/science/article/pii/S0012821X14000995

start_year=1999
end_year=2008
traces = pd.read_csv(f"events_traces_catalog/{start_year}_{end_year}_ok_picked_traces.csv")

EQ_ID = os.listdir(f"./tracer_demo/{start_year}_{end_year}_output")

traces["p_arrival_abs_time"] = pd.to_datetime(
    traces[["year", "month", "day", "hour", "minute", "second"]]
)

colnames = [
    "evt_lon",
    "evt_lat",
    "evt_depth",
    "sta_lon",
    "sta_lat",
    "sta_elev",
    "p_arrival",
    "s_arrival",
]
for eq in EQ_ID:
    event_file_path = f"./tracer_demo/{start_year}_{end_year}_output/{eq}/output.table"
    tracer_output = pd.read_csv(
        event_file_path, sep=r"\s+", names=colnames, header=None
    )
    trace_index = traces[traces["EQ_ID"] == int(eq)].index
    p_arrival = pd.to_timedelta(tracer_output["p_arrival"], unit="s")
    p_arrival.index = trace_index
    traces.loc[trace_index, "p_arrival_abs_time"] = (
        traces.loc[trace_index, "p_arrival_abs_time"] + p_arrival
    )
# traces 和 event 須將 eq_id: 29363 剔除 (velocity model calculate out of range)
final_traces = traces[traces["EQ_ID"] != 29363]
event = pd.read_csv(f"./events_traces_catalog/{start_year}_{end_year}_ok_events.csv")
final_event = event[event["EQ_ID"] != 29363]
# save catalog
# final_traces.to_csv(
#     f"./events_traces_catalog/{start_year}_{end_year}_picked_traces_p_arrival_abstime.csv", index=False
# )
# final_event.to_csv(
#     f"./events_traces_catalog/{start_year}_{end_year}_ok_events_p_arrival_abstime.csv", index=False
# )