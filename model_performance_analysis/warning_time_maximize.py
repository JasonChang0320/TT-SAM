import numpy as np
import pandas as pd
from analysis import Rolling_Warning, Warning_Time_Plotter

eq_id = 24784.0
label_type = "pga"
if label_type == "pga":
    label_threshold = np.log10(0.25)
    intensity = "IV"
if label_type == "pgv":
    label_threshold = np.log10(0.15)
    intensity = "V"

path = "../predict/station_blind_Vs30_bias2closed_station_2016"
output_path = f"{path}/mag bigger 5.5 predict"

prediction3_with_info = pd.read_csv(f"{path}/3 sec model11 with all info.csv")
prediction5_with_info = pd.read_csv(f"{path}/5 sec model11 with all info.csv")
prediction7_with_info = pd.read_csv(f"{path}/7 sec model11 with all info.csv")
prediction10_with_info = pd.read_csv(f"{path}/10 sec model11 with all info.csv")

rw_instance = Rolling_Warning(label_type="pga")
warning_df_with_station_info = (
    rw_instance.calculate_warning_time_at_different_issue_timing(
        prediction_in_different_timing=[
            prediction3_with_info,
            prediction5_with_info,
            prediction7_with_info,
            prediction10_with_info,
        ],
        time_list=[3, 5, 7, 10],
        event_filter="magnitude>=5",
    )
)

fig, ax = rw_instance.plot_maximum_warning_time(
    warning_df_with_station_info=warning_df_with_station_info,
    time_list=["3 second", "5 second", "7 second", "10 second"],
)
# fig.savefig(f"{path}/update warning_epi_vs_lead_time_mag_bigger_than_5.png",dpi=300)

event_info = warning_df_with_station_info[
    warning_df_with_station_info["EQ_ID"] == eq_id
]
fig,ax=rw_instance.plot_event_warning_time_with_distance_range(
    event_info=event_info, distance_range=[20, 60], event_loc=[120.543833333333, 22.922]
)

maximum_warning_time = warning_df_with_station_info["max_warning_time"]
maximum_warning_time = maximum_warning_time[maximum_warning_time > 0]
describe = maximum_warning_time.describe()
count = int(describe["count"])
mean = np.round(describe["mean"], 2)
std = np.round(describe["std"], 2)
median = np.round(describe["50%"], 2)
max = np.round(describe["max"], 2)
statistical_dict = rw_instance.calculate_statistical_value(warning_df_with_station_info)

fig, ax = rw_instance.plot_maximum_warning_time_histogram(
    warning_df_with_station_info,
    statistical_dict,
    title="Warning time in 2016 events magnitude >=5",
)
# fig.savefig(f"{output_path}/maximum warning time, magnitude bigger than 5.png",dpi=300)

single_event_statistical_dict = rw_instance.calculate_statistical_value(
    warning_df_with_station_info, filter=f"EQ_ID=={eq_id}"
)

fig, ax = rw_instance.plot_maximum_warning_time_histogram(
    warning_df_with_station_info,
    single_event_statistical_dict,
    filter=f"EQ_ID=={eq_id}",
    title=f"EQ ID: {eq_id}, Maximum warning time",
)

# fig.savefig(f"{output_path}/EQ ID{eq_id}, maximum warning time.png", dpi=300)
for sec, events_prediction in zip(
    [3, 5, 7, 10],
    [
        prediction3_with_info,
        prediction5_with_info,
        prediction7_with_info,
        prediction10_with_info,
    ],
):
    single_event_prediction = events_prediction.query(f"EQ_ID=={eq_id}")
    fig, ax = Warning_Time_Plotter.p_wave_pga_travel_time(
        event_prediction=single_event_prediction,
        title=f"EQ ID: {eq_id} {sec} sec prediction with p-wave and pga travel time",
    )
