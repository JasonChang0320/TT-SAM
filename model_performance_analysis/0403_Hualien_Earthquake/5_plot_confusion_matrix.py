import sys
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


sys.path.append("..")
from analysis import Precision_Recall_Factory

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

#################
label_threshold = np.log10(np.array([0.25]))
predict_label = np.array(max_prediction[f"max_predict"])
real_label = np.array(max_prediction["PGA"])
predict_logic = np.where(predict_label > label_threshold, 1, 0)
real_logic = np.where(real_label > label_threshold, 1, 0)
matrix = confusion_matrix(real_logic, predict_logic, labels=[1, 0])
accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)  # (TP+TN)/all
precision = matrix[0][0] / np.sum(matrix, axis=0)[0]  # TP/(TP+FP)
recall = matrix[0][0] / np.sum(matrix, axis=1)[0]

intensity = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
max_prediction["predicted_intensity"] = max_prediction["max_predict"].apply(
    Precision_Recall_Factory.pga_to_intensity
)
max_prediction["answer_intensity"] = max_prediction["PGA"].apply(Precision_Recall_Factory.pga_to_intensity)

intensity_confusion_matrix = confusion_matrix(
    max_prediction["answer_intensity"],
    max_prediction["predicted_intensity"],
    labels=intensity,
)

fig, ax = Precision_Recall_Factory.plot_intensity_confusion_matrix(
    intensity_confusion_matrix
)
# fig.savefig("confusion_matrix.png", dpi=300)
