import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import os

path = "predict/station_blind_Vs30_bias2closed_station_2016"
output_path = f"{path}/model11 confusion matrix"
if not os.path.isdir(output_path):
    os.mkdir(output_path)
label = "pga"
if label == "pga":
    Label_threshold = np.log10(
        np.array([0.080, 0.250, 0.80])  # 3,4,5級
    )  # g*9.8 = m/s^2
    unit = "m/s^2"
if label == "pgv":
    Label_threshold = np.log10(np.array([0.019, 0.057, 0.15]))  # 3,4,5級
    unit = "m/s"


def pga_to_intensity(value):
    pga_threshold = np.log10([0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10])
    intensity = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    for i, threshold in enumerate(pga_threshold):
        if value < threshold:
            return intensity[i]
    return intensity[-1]


intensity_score_dict = {"second": [], "intensity_score": []}
for mask_after_sec in [3, 5, 7, 10]:
    data = pd.read_csv(f"{path}/{mask_after_sec} sec model11 with all info.csv")

    predict_label = data["predict"]
    real_label = data["answer"]
    # calculate intensity score
    intensity = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    data["predicted_intensity"] = predict_label.apply(pga_to_intensity)
    data["answer_intensity"] = real_label.apply(pga_to_intensity)
    intensity_score = (
        (data["predicted_intensity"] == data["answer_intensity"]).sum()
    ) / len(data)
    intensity_score_dict["second"].append(mask_after_sec)
    intensity_score_dict["intensity_score"].append(intensity_score)
    intensity_table = pd.DataFrame(intensity_score_dict)

    intensity_table.to_csv(
        f"{output_path}/intensity table.csv",
        index=False,
    )
    # plot intensity score confusion matrix

    intensity_confusion_matrix = confusion_matrix(
        data["answer_intensity"], data["predicted_intensity"], labels=intensity
    )

    sn.set(rc={"figure.figsize": (8, 8)}, font_scale=1.2)  # for label size
    fig, ax = plt.subplots()
    sn.heatmap(
        intensity_confusion_matrix,
        ax=ax,
        xticklabels=intensity,
        yticklabels=intensity,
        fmt="g",
        annot=True,
        annot_kws={"size": 16},
        cmap="Reds",
        cbar=True,
        cbar_kws={"label": "number of traces"},
    )  # font size
    ax.set_xlabel("Predicted intensity")
    ax.set_ylabel("Actual intensity")
    ax.set_title(
        f"{mask_after_sec} sec intensity confusion matrix, intensity score: {np.round(intensity_score,3)}"
    )
    fig.savefig(
        f"{output_path}/{mask_after_sec} sec intensity confusion matrix, {label} intensity threshold.png",
        dpi=300,
    )

    performance_score = {
        f"{label}_threshold ({unit})": [],
        "confusion matrix": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
    }
    for label_threshold in Label_threshold:
        predict_logic = np.where(predict_label > label_threshold, 1, 0)
        real_logic = np.where(real_label > label_threshold, 1, 0)

        matrix = confusion_matrix(real_logic, predict_logic, labels=[1, 0])
        accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)  # (TP+TN)/all
        precision = matrix[0][0] / np.sum(matrix, axis=0)[0]  # TP/(TP+FP)
        recall = matrix[0][0] / np.sum(matrix, axis=1)[0]  # TP/(TP+FP)
        F1_score = 2 / ((1 / precision) + (1 / recall))
        performance_score[f"{label}_threshold ({unit})"].append(
            np.round((10**label_threshold), 3)
        )  # m/s^2 / 9.8 = g
        performance_score["confusion matrix"].append(matrix)
        performance_score["accuracy"].append(accuracy)
        performance_score["precision"].append(precision)
        performance_score["recall"].append(recall)
        performance_score["f1_score"].append(F1_score)
        sn.set(rc={"figure.figsize": (8, 5)}, font_scale=1.2)  # for label size
        fig, ax = plt.subplots()
        sn.heatmap(
            matrix,
            ax=ax,
            xticklabels=["Predict True", "Predict False"],
            yticklabels=["Actual True", "Actual False"],
            fmt="g",
            annot=True,
            annot_kws={"size": 16},
            cmap="Reds",
        )  # font size
        ax.set_title(
            f"EEW {mask_after_sec} sec confusion matrix, {label} threshold: {np.round((10**label_threshold),3)} ({unit})"
        )
        # fig.savefig(
        #     f"{output_path}/{mask_after_sec} sec {label}_threshold {np.round((10**label_threshold),3)}.png"
        # ,dpi=300)
        predict_table = pd.DataFrame(performance_score)
        # predict_table.to_csv(
        #     f"{output_path}/{mask_after_sec} sec confusion matrix table.csv",
        #     index=False,
        # )
    # label threshold performance
    axis_fontsize = 30
    Fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, column in enumerate(["precision", "recall", "f1_score"]):
        x_axis = [
            str(label) for label in performance_score[f"{label}_threshold ({unit})"]
        ]
        axes[i].bar(x_axis, performance_score[f"{column}"])
        axes[i].set_title(f"{column}", fontsize=axis_fontsize)
        axes[i].tick_params(axis="both", which="major", labelsize=axis_fontsize - 15)
        axes[i].set_ylim(0, 1)
    axes[1].set_xlabel(f"{label} threshold ({unit})", fontsize=axis_fontsize - 12)
    # Fig.savefig(f"{output_path}/{mask_after_sec} sec F1 score.png",dpi=300)

# pga threshold by time plot
sec3_table = pd.read_csv(f"{output_path}/3 sec confusion matrix table.csv")
sec5_table = pd.read_csv(f"{output_path}/5 sec confusion matrix table.csv")
sec7_table = pd.read_csv(f"{output_path}/7 sec confusion matrix table.csv")
sec10_table = pd.read_csv(f"{output_path}/10 sec confusion matrix table.csv")

if label == "pga":
    Label_threshold = [0.080, 0.250, 0.80]
if label == "pgv":
    Label_threshold = [0.019, 0.057, 0.15]
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
plot_index = [0, 1, 2]
for index, label_threshold in zip(plot_index, Label_threshold):
    Accuracy = []
    Precision = []
    Recall = []
    F1_score = []
    for table in [sec3_table, sec5_table, sec7_table, sec10_table]:
        tmp_table = table[table[f"{label}_threshold ({unit})"] == label_threshold]
        Accuracy.append(tmp_table["accuracy"].values[0])
        Precision.append(tmp_table["precision"].values[0])
        Recall.append(tmp_table["recall"].values[0])
        F1_score.append(tmp_table["f1_score"].values[0])
    ax[index].plot(Accuracy, marker="o", label="accuracy")
    ax[index].plot(Precision, marker="o", label="precision")
    ax[index].plot(Recall, marker="o", label="recall")
    ax[index].plot(F1_score, marker="o", label="F1 score")
    ax[index].set_title(f"{label} threshold: {label_threshold} (m/s^2)", fontsize=25)
    ax[index].set_xticks([0, 1, 2, 3])
    ax[index].set_xticklabels(["3", "5", "7", "10"])
    ax[index].set_xlabel("After first triggered station (sec)", fontsize=15)
    ax[index].set_ylim(-0.1, 1.1)
    ax[index].legend()
# fig.savefig(
#     f"{output_path}/different {label} threshold performance by time.png"
# ,dpi=300)
