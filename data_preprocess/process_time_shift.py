import pandas as pd

"""
The script calculates the time between first triggered station got waveform and earthquake occurred.
"""

input_path = "./events_traces_catalog"
catalog = pd.read_csv(f"{input_path}/1999_2019_final_catalog.csv")
traces = pd.read_csv(f"{input_path}/1999_2019_final_traces_Vs30.csv")

traces.loc[traces.index, "p_arrival_abs_time"] = pd.to_datetime(
    traces["p_arrival_abs_time"], format="%Y-%m-%d %H:%M:%S"
)
catalog["event_time"] = pd.to_datetime(
    catalog[["year", "month", "day", "hour", "minute", "second"]]
)


eq_id_list = [24757, 24784, 25112, 25193, 25225, 25396, 25401, 25561, 25900]

for eq_id in eq_id_list:
    event = catalog.query(f"EQ_ID=={eq_id}")
    triggered_trace = traces.query(f"EQ_ID=={eq_id}")

    first_triggered_trace = triggered_trace.loc[
        triggered_trace["p_arrival_abs_time"].idxmin()
    ]
    p_wave_propogated_time = (
        first_triggered_trace["p_arrival_abs_time"] - event["event_time"]
    )
    print(eq_id, p_wave_propogated_time)
