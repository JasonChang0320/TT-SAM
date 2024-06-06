import math
import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics import confusion_matrix
from analysis import Intensity_Plotter


def haversine(lat1, lon1, lat2, lon2):
    # 將經緯度轉換為弧度
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 地球半徑（千米）
    radius = 6371

    # 計算距離
    distance = radius * c

    return distance


# EEW calculate intensity
path = "../CWA_EEW_report"
site_info = pd.read_excel(f"{path}/site.xlsx")
catalog = pd.read_excel(f"{path}/EEW2016.xlsx")
catalog.columns = [
    "event_time",
    "catalog_lon",
    "catalog_lat",
    "catalog_mag",
    "catalog_dep",
    "eew_lon",
    "eew_lat",
    "eew_mag",
    "eew_dep",
    "eew_time",
]
catalog = catalog.query("catalog_mag>=5.5")
catalog["event_time"] = catalog["event_time"].astype(str)
catalog.dropna(inplace=True)
catalog.reset_index(drop=True, inplace=True)
predict_dict = {
    "event_time": [],
    "sta_lat": [],
    "sta_lon": [],
    "predict_pga": [],
    "station_code": [],
    "process_time": [],
}
for i in range(len(catalog)):
    print(catalog["event_time"][i])
    lat = catalog["eew_lat"][i]
    lon = catalog["eew_lon"][i]
    dep = catalog["eew_dep"][i]
    mag = catalog["eew_mag"][i]
    for j in range(len(site_info)):
        if dep < 40:
            Si = site_info["site_s"][j]
            hypo_dist = math.sqrt(
                math.pow(
                    haversine(lat, lon, site_info["lat"][j], site_info["lon"][j]), 2
                )
                + math.pow(dep, 2)
            )
            pga = (
                12.44 * math.exp(1.31 * mag) * math.pow(hypo_dist, -1.837) * Si
            )  # 2021_0303 from Hsiao
        else:
            Si = site_info["site_d"][j]
            hypo_dist = math.sqrt(
                math.pow(
                    haversine(lat, lon, site_info["lat"][j], site_info["lon"][j]), 2
                )
                + math.pow(dep, 2)
            )
            pga = (
                12.44 * math.exp(1.31 * mag) * math.pow(hypo_dist, -1.837) * Si
            )  # 2021_0303 from Hsiao

        predict_dict["event_time"].append(catalog["event_time"][i])
        predict_dict["sta_lat"].append(site_info["lat"][j])
        predict_dict["sta_lon"].append(site_info["lon"][j])
        predict_dict["predict_pga"].append(pga)
        predict_dict["station_code"].append(site_info["code"][j])
        predict_dict["process_time"].append(catalog["eew_time"][i])

predict_df = pd.DataFrame(predict_dict)
predict_df["event_time"] = predict_df["event_time"].astype(str)
# merge ground true pga


pattern = r"[=,]"
true_pga_dict = {
    "event_time": [],
    "station_code": [],
    "sta_lon": [],
    "sta_lat": [],
    "dist": [],
    "PGA(V)": [],
    "PGA(NS)": [],
    "PGA(EW)": [],
}
files = os.listdir(f"{path}/event_true_pga")
for file in files:
    with open(f"{path}/event_true_pga/{file}", "r", encoding="iso-8859-1") as event:
        start_line = 5
        lines = event.readlines()
        for i in range(start_line, len(lines)):
            line = lines[i]
            result = re.split(pattern, line.strip())
            true_pga_dict["event_time"].append(file.replace(".txt", ""))
            true_pga_dict["station_code"].append(result[1].replace(" ", ""))
            true_pga_dict["sta_lon"].append(float(result[5].replace(" ", "")))
            true_pga_dict["sta_lat"].append(float(result[7].replace(" ", "")))
            true_pga_dict["dist"].append(float(result[9].replace(" ", "")))
            true_pga_dict["PGA(V)"].append(float(result[13].replace(" ", "")))
            true_pga_dict["PGA(NS)"].append(float(result[15].replace(" ", "")))
            true_pga_dict["PGA(EW)"].append(float(result[17].replace(" ", "")))

true_pga_df = pd.DataFrame(true_pga_dict)

final_table = pd.merge(
    predict_df,
    true_pga_df,
    on=["event_time", "station_code"],
    how="left",
    suffixes=["_pre", "_true"],
)
final_table.dropna(inplace=True)
time_eqid_dict = {
    "eqid": [24757, 24784, 25112, 25193, 25225, 25396, 25401, 25561, 25900],
    "event_time": [
        "201601190213026",
        "201602051957026",
        "201604110545009",
        "201604271517014",
        "201604271819006",
        "201605120317015",
        "201605120429055",
        "201605310523046",
        "201610061552000",
    ],
}
time_eqid_df = pd.DataFrame(time_eqid_dict)
final_traces = pd.merge(time_eqid_df, final_table, on="event_time", how="right")
final_catalog = pd.merge(time_eqid_df, catalog, on="event_time", how="left")
# final_traces.to_csv("cwa_test_eew_events.csv",index=False)
# final_traces.to_csv("cwa_test_eew_traces.csv",index=False)
# =========calculate residual mean and std
final_traces["PGA"] = np.sqrt(
    final_traces["PGA(V)"] ** 2
    + final_traces["PGA(NS)"] ** 2
    + final_traces["PGA(EW)"] ** 2
)
residual_mean = (
    np.log10(final_traces["predict_pga"] * 0.01) - np.log10(final_traces["PGA"] * 0.01)
).mean()
residual_std = (
    np.log10(final_traces["predict_pga"] * 0.01) - np.log10(final_traces["PGA"] * 0.01)
).std()
label_threshold = np.log10(np.array([0.250]))  # 3,4,5級
predict_logic = np.where(
    np.log10(final_traces["predict_pga"] * 0.01) > label_threshold[0], 1, 0
)
real_logic = np.where(np.log10(final_traces["PGA"] * 0.01) > label_threshold[0], 1, 0)

matrix = confusion_matrix(real_logic, predict_logic, labels=[1, 0])
accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)  # (TP+TN)/all
precision = matrix[0][0] / np.sum(matrix, axis=0)[0]  # TP/(TP+FP)
recall = matrix[0][0] / np.sum(matrix, axis=1)[0]  # TP/(TP+FP)
F1_score = 2 / ((1 / precision) + (1 / recall))

fig, ax = Intensity_Plotter.plot_true_predicted(
    y_true=np.log10(final_traces["PGA"] * 0.01),
    y_pred=np.log10(final_traces["predict_pga"] * 0.01),
    quantile=False,
    agg="point",
    point_size=70,
    target="pga",
    title=f"CWA EEW prediction in 2016 M>5.5 events",
)
# fig.savefig("CWA EEW report/true predict plot.png",dpi=300)
# =========plot intensity map

for eqid in final_catalog["eqid"]:
    label_type = "pga"
    fig, ax = Intensity_Plotter.plot_CWA_EEW_intensity_map(
        final_traces, final_catalog, eqid, label_type
    )

    # fig.savefig(f"paper image/eqid_{eqid}_CWA_eew_report.pdf",dpi=300)
