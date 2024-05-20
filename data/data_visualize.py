import numpy as np
import pandas as pd
from visualize import Plot_Train_Test_Data, Increase_High_Data_Test


data_path = "../data_preprocess/events_traces_catalog"
origin_catalog = pd.read_csv(f"{data_path}/1999_2019_final_catalog.csv")
traces_catalog = pd.read_csv(f"{data_path}/1999_2019_final_traces_Vs30.csv")
test_year = 2016
train_catalog = origin_catalog.query(f"year!={test_year}")
test_catalog = origin_catalog.query(f"year=={test_year}")
# events histogram
fig, ax = Plot_Train_Test_Data.event_histogram(
    train_catalog, test_catalog, key="magnitude", xlabel="magnitude"
)
# fig.savefig(f"paper image/event depth distribution.png",dpi=300)
# fig.savefig(f"paper image/event depth distribution.pdf",dpi=300)

# event distribution in map
fig, ax = Plot_Train_Test_Data.event_map(train_catalog, test_catalog)
# fig.savefig(f"paper image/event distribution map.png",dpi=300)
# fig.savefig(f"paper image/event distribution map.pdf",dpi=300)

# traces pga histogram
fig, ax = Plot_Train_Test_Data.pga_histogram(traces_catalog, test_year=test_year)
# fig.savefig(f"paper image/trace pga distribution.png",dpi=300)
# fig.savefig(f"paper image/trace pga distribution.pdf",dpi=300)


# test oversampling method
data_path = "./TSMIP_1999_2019_Vs30.hdf5"
origin_PGA = Increase_High_Data_Test.load_dataset_into_list(
    data_path, oversample_rate=1, bias_to_close_station=False
)
oversampled_PGA = Increase_High_Data_Test.load_dataset_into_list(
    data_path, oversample_rate=1.5, bias_to_close_station=False
)

bias_closed_sta_PGA = Increase_High_Data_Test.load_dataset_into_list(
    data_path, oversample_rate=1.5, bias_to_close_station=True
)

origin_PGA_array = np.array(origin_PGA)
origin_high_intensity_rate = np.sum(origin_PGA_array > np.log10(0.250)) / len(
    origin_PGA_array
)
print(f"origin rate:{origin_high_intensity_rate}")

oversampled_PGA_array = np.array(oversampled_PGA)
oversampled_high_intensity_rate = np.sum(oversampled_PGA_array > np.log10(0.250)) / len(
    oversampled_PGA_array
)
print(f"oversampled rate:{oversampled_high_intensity_rate}")

bias_closed_sta_PGA_array = np.array(bias_closed_sta_PGA)
bias_closed_sta_high_intensity_rate = np.sum(
    bias_closed_sta_PGA_array > np.log10(0.250)
) / len(bias_closed_sta_PGA_array)
print(f"bias_closed_sta rate:{bias_closed_sta_high_intensity_rate}")

fig, ax = Increase_High_Data_Test.plot_pga_histogram(
    bias_closed_sta_PGA,
    oversampled_PGA,
    origin_PGA,
    origin_high_intensity_rate,
    oversampled_high_intensity_rate,
    bias_closed_sta_high_intensity_rate,
)
# fig.savefig("PGA distribution.png", dpi=300,bbox_inches='tight')
