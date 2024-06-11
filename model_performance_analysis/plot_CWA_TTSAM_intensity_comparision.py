import pandas as pd
from analysis import Intensity_Plotter

eqid = 24784
data_folder = "../data/station_information"
multi_station = pd.read_csv(f"{data_folder}/multi-station.txt", sep=" ")
station_dataset = pd.read_csv(f"{data_folder}/TSMIPstations_new.csv")
cwa_event = pd.read_csv(f"{data_folder}/cwa_test_eew_events.csv")
cwa_traces = pd.read_csv(f"{data_folder}/cwa_test_eew_traces.csv")
process_time = int(cwa_event.query(f"eqid=={eqid}")["eew_time"].values[0])
event_lat = cwa_event.query(f"eqid=={eqid}")["catalog_lat"].values[0]
event_lon = cwa_event.query(f"eqid=={eqid}")["catalog_lon"].values[0]
mag = cwa_event.query(f"eqid=={eqid}")["catalog_mag"].values[0]
merge_data = pd.merge(
    multi_station,
    station_dataset[["station_code", "location_code"]],
    left_on="TSMIP",
    right_on="station_code",
    how="left",
)


cwa_merge_data = pd.merge(
    cwa_traces[
        ["eqid", "predict_pga", "station_code", "sta_lat_pre", "sta_lon_pre", "PGA"]
    ],
    merge_data[["CWASN", "location_code"]],
    left_on="station_code",
    right_on="CWASN",
    how="inner",
)

tt_sam = pd.read_csv(
    "../predict/station_blind_Vs30_bias2closed_station_2016/7 sec model11 with all info.csv"
)

ttsam_merge_data = pd.merge(
    tt_sam[["EQ_ID", "predict", "answer", "latitude", "longitude", "station_name"]],
    merge_data[["CWASN", "location_code"]],
    left_on="station_name",
    right_on="location_code",
    how="inner",
)
ttsam_merge_data["sta_lon_pre"] = ttsam_merge_data["longitude"]
ttsam_merge_data["sta_lat_pre"] = ttsam_merge_data["latitude"]
ttsam_merge_data["predict_pga"] = (10 ** ttsam_merge_data["predict"]) * 100
ttsam_merge_data["observed_pga"] = (10 ** ttsam_merge_data["answer"]) * 100
ttsam_merge_data["eqid"] = ttsam_merge_data["EQ_ID"]
# ==============================

# change "ttsam_merge_data" or "cwa_merge_data" to plot each system intensity map
event = ttsam_merge_data.query(f"eqid=={eqid}")

fig, ax = Intensity_Plotter.plot_intensity_scatter_map(
    event,
    event_lon,
    event_lat,
    mag,
    pga_column="observed_pga",
    title="Observed intensity",
)
# fig.savefig(f"../CWA_EEW_report/eqid_{eqid}_intensity.png", dpi=300)