import matplotlib.pyplot as plt

plt.subplots()
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os
from scipy.ndimage import zoom
import sys
sys.path.append("..")
from data.multiple_sta_dataset import multiple_station_dataset
from model.CNN_Transformer_Mixtureoutput_TEAM import CNN_feature_map
import os
from scipy.signal import hilbert
from tlcc_analysis import Plotter,Calculator


mask_after_sec = 10
sample_rate = 200
label = "pga"
data = multiple_station_dataset(
    "../data/TSMIP_1999_2019_Vs30.hdf5",
    mode="test",
    mask_waveform_sec=mask_after_sec,
    test_year=2016,
    label_key=label,
    mag_threshold=0,
    input_type="acc",
    data_length_sec=15,
)
# need station name
output_path = "../predict/station_blind_Vs30_bias2closed_station_2016"
predict = pd.read_csv(f"{output_path}/{mask_after_sec} sec model11 with all info.csv")

# ===========prepare model==============
device = torch.device("cuda")
num = 11
path = f"../model/model{num}.pt"
emb_dim = 150
mlp_dims = (150, 100, 50, 30, 10)
CNN_model = CNN_feature_map(mlp_input=5665).cuda()

full_model_parameter = torch.load(path)
# ===========load CNN parameter==============
CNN_parameter = {}
for name, param in full_model_parameter.items():
    if (
        "model_CNN" in name
    ):  # model_CNN.conv2d1.0.weight : conv2d1.0.weight didn't match
        name = name.replace("model_CNN.", "")
        CNN_parameter[name] = param
CNN_model.load_state_dict(CNN_parameter)

event_index_list = []
for eq_id in data.events_index:
    event_index_list.append(eq_id[0][0, 0])

eq_first_index = Calculator.first_occurrences_indices(event_index_list)

# plot feature map and calculate correlation
attribute_dict = {
    "euclidean_norm": {"correlation": [], "tlcc_max_correlation": [], "max_delay": []},
    "vertical_envelope": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "NS_envelope": {"correlation": [], "tlcc_max_correlation": [], "max_delay": []},
    "EW_envelope": {"correlation": [], "tlcc_max_correlation": [], "max_delay": []},
    "vertical_instantaneous_phase": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "NS_instantaneous_phase": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "EW_instantaneous_phase": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "vertical_instantaneous_freq": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "NS_instantaneous_freq": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "EW_instantaneous_freq": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
}
print(len(eq_first_index.keys()))
for key, index in tqdm(zip(eq_first_index.keys(), eq_first_index.values())):
    event_output_path = (
        f"{output_path}/{mask_after_sec} sec cnn feature map/each event/{str(key)}"
    )
    if not os.path.isdir(f"{event_output_path}"):
        os.makedirs(f"{event_output_path}")
    sample = data[index]
    waveform = sample["waveform"]

    not_padding_station_number = (
        (torch.from_numpy(sample["sta"]) != 0).all(dim=1).sum().item()
    )
    single_event_prediction = predict.query(f"EQ_ID=={key}")
    input_station_list = single_event_prediction["station_name"][
        :not_padding_station_number
    ].tolist()
    if len(input_station_list) < 25:
        input_station_list += [np.nan] * (25 - len(input_station_list))

    p_picks = sample["p_picks"].flatten().tolist()
    # plot 24784 input waveform
    if key == 24784:
        for i in range(not_padding_station_number):
            single_waveform=waveform[i]
            input_station=input_station_list[i]
            fig, ax = Plotter.plot_waveform(single_waveform, key, input_station,index=i)

    cnn_input = torch.DoubleTensor(waveform).float().cuda()
    cnn_output, layer_output = CNN_model(cnn_input)
    numeric_array = np.array(layer_output[-1].detach().cpu(), dtype=np.float32)
    feature_map = np.mean(numeric_array, axis=1)
    scale_factor_h = waveform.shape[0] / feature_map.shape[0]
    scale_factor_w = waveform.shape[1] / feature_map.shape[1]

    # zoom out feature map
    resized_feature_map = zoom(feature_map, (scale_factor_h, scale_factor_w), order=3)
    component_dict = {}
    euclidean_waveform = np.linalg.norm(waveform, axis=2) / np.sqrt(3)
    component_dict[f"euclidean_norm"] = euclidean_waveform
    for com, component in enumerate(["vertical", "NS", "EW"]):
        analytic_signal = hilbert(waveform[:, :, com])
        envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.abs(
            (np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate)
        )
        component_dict[f"{component}_envelope"] = envelope
        component_dict[f"{component}_instantaneous_phase"] = instantaneous_phase
        component_dict[f"{component}_instantaneous_freq"] = instantaneous_frequency

    for attribute in component_dict: #calculate correlation to different attribute
        for i in range(not_padding_station_number):
            correlation_starttime = p_picks[i] - sample_rate
            correlation_endtime = p_picks[0] + (mask_after_sec + 1) * sample_rate
            if mask_after_sec == 10:
                correlation_endtime = p_picks[0] + (mask_after_sec) * sample_rate
            try:
                correlation = np.corrcoef(
                    component_dict[attribute][
                        i, correlation_starttime:correlation_endtime
                    ],
                    resized_feature_map[i, correlation_starttime:correlation_endtime],
                )[0, 1]
                delay_values, tlcc_values = Calculator.calculate_tlcc(
                    component_dict[attribute][
                        i, correlation_starttime:correlation_endtime
                    ],
                    resized_feature_map[i, correlation_starttime:correlation_endtime],
                    max_delay=100,
                )
            except:  # second=10 case
                correlation = np.corrcoef(
                    component_dict[attribute][
                        i, correlation_starttime:correlation_endtime
                    ],
                    resized_feature_map[
                        i, correlation_starttime + 1 : correlation_endtime
                    ],
                )[0, 1]
                delay_values, tlcc_values = Calculator.calculate_tlcc(
                    component_dict[attribute][
                        i, correlation_starttime:correlation_endtime
                    ],
                    resized_feature_map[
                        i, correlation_starttime + 1 : correlation_endtime
                    ],
                    max_delay=100,
                )
            attribute_dict[attribute]["correlation"].append(correlation)
            max_index = np.argmax(tlcc_values)
            max_correlation = tlcc_values[max_index]
            max_delay = delay_values[max_index]
            attribute_dict[attribute]["tlcc_max_correlation"].append(max_correlation)
            attribute_dict[attribute]["max_delay"].append(max_delay)

            if key == 24784:  # plot
                fig, ax = Plotter.plot_correlation_curve_with_shift_time(
                    delay_values, tlcc_values, key, attribute,index=i,mask_after_sec=mask_after_sec, output_path=None
                )
                attribute_arr=Calculator.normalize_to_zero_one(component_dict[attribute][i])
                resized_feature_map=Calculator.normalize_to_zero_one(resized_feature_map[i])
                fig, ax = Plotter.plot_attribute_with_feature_map(
                    attribute_arr,
                    resized_feature_map,
                    key,
                    attribute,
                    correlation_starttime,
                    correlation_endtime,
                    correlation,
                    tlcc_values,
                    input_station_list[i],
                )

output_path = f"{output_path}/{mask_after_sec} sec cnn feature map"

for attribute in attribute_dict: #statistical analysis
    TLCC_mean = np.round(
        np.array(attribute_dict[attribute]["tlcc_max_correlation"]).mean(), 2
    )
    TLCC_std = np.round(
        np.array(attribute_dict[attribute]["tlcc_max_correlation"]).std(), 2
    )
    fig, ax = Plotter.plot_correlation_hist(
        attribute_dict, attribute, TLCC_mean, TLCC_std, mask_after_sec, output_path=None
    )
    # x: time sample lag, y: max correlation (TLCC)
    fig, ax = Plotter.plot_time_shifted_with_correlation(
        attribute_dict, attribute, TLCC_mean, TLCC_std, mask_after_sec, output_path=None
    )
    # max correlation time delay hist
    delay_mean = np.round(np.array(attribute_dict[attribute]["max_delay"]).mean(), 2)
    delay_std = np.round(np.array(attribute_dict[attribute]["max_delay"]).std(), 2)
    fig, ax = Plotter.plot_time_shifted_with_hist(
        attribute_dict,
        attribute,
        delay_mean,
        delay_std,
        mask_after_sec,
        output_path=None,
    )

# belowed data is correlation with attributes in different seconds
data = np.array(
    [
        [0.61, 0.53, 0.49, 0.46],
        [0.68, 0.58, 0.52, 0.5],
        [0.59, 0.51, 0.47, 0.46],
        [0.58, 0.5, 0.47, 0.45],
        [0.29, 0.23, 0.18, 0.12],
        [0.29, 0.23, 0.18, 0.12],
        [0.29, 0.23, 0.18, 0.12],
        [0.3, 0.22, 0.16, 0.11],
        [0.29, 0.21, 0.15, 0.1],
        [0.3, 0.21, 0.15, 0.1],
    ]
)
attributes = [
    "Euclidean norm",
    "Vertical envelope",
    "NS envelope",
    "EW envelope",
    "Vertical phase",
    "NS phase",
    "EW phase",
    "Vertical frequency",
    "NS frequency",
    "EW frequency",
]
output_path = "./predict/station_blind_Vs30_bias2closed_station_2016"
fig, ax = Plotter.correlation_with_attributes_heat_map(data, attributes, output_path=None)
