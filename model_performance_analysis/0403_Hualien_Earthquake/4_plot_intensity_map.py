import sys
import pandas as pd

sys.path.append("..")
from analysis import Intensity_Plotter

mask_sec = 3
event_lon = 121.66
event_lat = 23.77
magnitude = 7.2
answer = pd.read_csv(f"true_answer.csv")

# merge 3 5 7 10 sec to find maximum predicted pga
prediction_3 = pd.read_csv(f"no_include_broken_data_prediction/3_sec_prediction.csv")
prediction_5 = pd.read_csv(f"no_include_broken_data_prediction/5_sec_prediction.csv")
prediction_7 = pd.read_csv(f"no_include_broken_data_prediction/7_sec_prediction.csv")
prediction_10 = pd.read_csv(f"no_include_broken_data_prediction/10_sec_prediction.csv")

max_prediction = pd.concat(
    [
        prediction_3,
        prediction_5["predict"],
        prediction_7["predict"],
        prediction_10["predict"],
    ],
    axis=1,
)

max_prediction.columns = [
    "3_predict",
    "station_name",
    "latitude",
    "longitude",
    "elevation",
    "5_predict",
    "7_predict",
    "10_predict",
]
max_prediction["max_predict"] = max_prediction.apply(
    lambda row: max(
        row["3_predict"], row["5_predict"], row["7_predict"], row["10_predict"]
    ),
    axis=1,
)

max_prediction = pd.merge(
    answer, max_prediction, how="left", left_on="location_code", right_on="station_name"
)
max_prediction.dropna(inplace=True)

eventmeta = pd.DataFrame(
    {"longitude": [event_lon], "latitude": [event_lat], "magnitude": [magnitude]}
)

Intensity_Plotter.plot_intensity_map(
    trace_info=max_prediction,
    eventmeta=eventmeta,
    label_type="pga",
    true_label=max_prediction["PGA"],
    pred_label=max_prediction[f"{mask_sec}_predict"],
    sec=mask_sec,
    min_epdis=10.87177078,  # 0.1087度轉成km
    EQ_ID=None,
    grid_method="linear",
    pad=100,
    title=f"{mask_sec} sec intensity Map",
)
# fig.savefig(f"true_intensity_map_without_broken_data/{mask_sec}_sec.png",dpi=300)
