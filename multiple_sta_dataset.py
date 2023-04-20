from itertools import repeat

import h5py
import numpy as np
import pandas as pd
import torch
from numpy import ma
from scipy.signal import butter, lfilter
from torch.utils.data import DataLoader, Dataset, Subset


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def shift_waveform(waveform, p_pick, start_before_sec=5, total_sec=30, sample_rate=200):
    start_point = p_pick - start_before_sec * sample_rate
    cut_waveform = waveform[start_point:, :]
    padded_waveform = np.pad(
        cut_waveform,
        ((0, total_sec * sample_rate - len(cut_waveform)), (0, 0)),
        "constant",
    )
    # ((top, bottom), (left, right))

    return padded_waveform


class intensity_classifier:
    def __init__(self, label=None):
        if label == "pga":
            self.threshold = np.log10([0.008, 0.025, 0.08, 0.25, 0.8, 2.5, 8])
            self.label = [0, 1, 2, 3, 4, 5, 6, 7]
        if label == "pgv":
            self.threshold = np.log10([0.002, 0.007, 0.019, 0.057, 0.15, 0.5, 1.4])
            self.label = [0, 1, 2, 3, 4, 5, 6, 7]

    def classify(self, input_array):
        output_array = np.zeros_like(input_array)
        for i in range(len(input_array)):
            if input_array[i] < self.threshold[0]:
                output_array[i] = self.label[0]
            elif input_array[i] < self.threshold[1]:
                output_array[i] = self.label[1]
            elif input_array[i] < self.threshold[2]:
                output_array[i] = self.label[2]
            elif input_array[i] < self.threshold[3]:
                output_array[i] = self.label[3]
            elif input_array[i] < self.threshold[4]:
                output_array[i] = self.label[4]
            elif input_array[i] < self.threshold[5]:
                output_array[i] = self.label[5]
            elif input_array[i] < self.threshold[6]:
                output_array[i] = self.label[6]
            elif input_array[i] >= self.threshold[6]:
                output_array[i] = self.label[7]
        return output_array


class multiple_station_dataset(Dataset):
    def __init__(
        self,
        data_path,
        sampling_rate=200,
        data_length_sec=30,
        test_year=2018,
        mode="train",
        limit=None,
        input_type="acc",
        label_key="pga",
        mask_waveform_sec=None,
        mask_waveform_random=False,
        dowmsampling=False,
        oversample=1,
        oversample_mag=4,
        max_station_num=25,
        label_target=25,
        sort_by_picks=True,
        oversample_by_labels=False,
        mag_threshold=0,
        weight_label=False,
    ):
        event_metadata = pd.read_hdf(data_path, "metadata/event_metadata")
        trace_metadata = pd.read_hdf(data_path, "metadata/traces_metadata")
        event_metadata = event_metadata[event_metadata["magnitude"] >= mag_threshold]
        if mode == "train":
            event_test_mask = [
                int(year) != test_year for year in event_metadata["year"]
            ]
            trace_test_mask = [
                int(year) != test_year for year in trace_metadata["year"]
            ]
            event_metadata = event_metadata[event_test_mask]
            trace_metadata = trace_metadata[trace_test_mask]
        elif mode == "test":
            event_test_mask = [
                int(year) == test_year for year in event_metadata["year"]
            ]
            trace_test_mask = [
                int(year) == test_year for year in trace_metadata["year"]
            ]
            event_metadata = event_metadata[event_test_mask]
            trace_metadata = trace_metadata[trace_test_mask]

        if limit:
            event_metadata = event_metadata.iloc[:limit]
        metadata = {}
        data = {}
        with h5py.File(data_path, "r") as f:
            decimate = 1

            skipped = 0
            contained = []
            events_index = np.zeros((1, 2), dtype=int)
            # events_index=[]
            for _, event in event_metadata.iterrows():
                event_name = str(int(event["EQ_ID"]))
                # print(event_name)
                if (
                    event_name not in f["data"]
                ):  # use catalog eventID campare to data eventID
                    skipped += 1
                    contained += [False]
                    continue
                contained += [True]
                g_event = f["data"][event_name]
                for key in g_event:
                    if key not in data:
                        data[key] = []
                    if key == f"{input_type}_traces":
                        index = np.arange(g_event[key].shape[0]).reshape(-1, 1)
                        eventid = (
                            np.array([str(event_name)] * g_event[key].shape[0])
                            .astype(np.int32)
                            .reshape(-1, 1)
                        )
                        single_event_index = np.concatenate([eventid, index], axis=1)
                    if key == label_key:
                        data[key] += [g_event[key][()]]
                    if key == "p_picks":
                        data[key] += [g_event[key][()]]
                    if key == "station_name":
                        data[key] += [g_event[key][()]]
                    if key == "p_picks":
                        data[key][-1] //= decimate

                events_index = np.append(events_index, single_event_index, axis=0)
                # events_index.append(single_event_index)
            events_index = np.delete(events_index, [0], 0)
        labels = np.concatenate(data[label_key], axis=0)
        stations = np.concatenate(data["station_name"], axis=0)
        picks = np.concatenate(data["p_picks"], axis=0)
        mask = (events_index != 0).any(axis=1)
        # label is nan mask
        mask = np.logical_and(mask, ~np.isnan(labels))
        # random delete too small labels
        if dowmsampling:
            small_labels_array = labels < np.log10(0.019)
            np.random.seed(0)
            random_array = np.random.choice(
                [True, False], size=small_labels_array.shape
            )
            random_delete_mask = np.logical_and(small_labels_array, random_array)
            mask = np.logical_and(mask, ~random_delete_mask)

        labels = np.expand_dims(np.expand_dims(labels, axis=1), axis=2)
        stations = np.expand_dims(np.expand_dims(stations, axis=1), axis=2)
        p_picks = picks[mask]
        stations = stations[mask]
        labels = labels[mask]

        ok_events_index = events_index[mask]
        ok_event_id = np.intersect1d(
            np.array(event_metadata["EQ_ID"].values), ok_events_index
        )
        if oversample > 1:
            oversampled_catalog = []
            filter = event_metadata["magnitude"] >= oversample_mag
            oversample_catalog = np.intersect1d(
                np.array(event_metadata[filter]["EQ_ID"].values), ok_events_index
            )
            for eventid in oversample_catalog:
                catch_mag = event_metadata["EQ_ID"] == eventid
                mag = event_metadata[catch_mag]["magnitude"]
                repeat_time = int(oversample ** (mag - 1) - 1)
                oversampled_catalog.extend(repeat(eventid, repeat_time))

            oversampled_catalog = np.array(oversampled_catalog)
            ok_event_id = np.concatenate([ok_event_id, oversampled_catalog])
        if oversample_by_labels:
            oversampled_labels = []
            oversampled_picks = []
            labels = labels.flatten()
            filter = labels > np.log10(0.057)
            oversample_events_index = ok_events_index[filter]
            oversample_p_pick = p_picks[filter]
            Repeat_time = 4.5 ** (1.5 ** labels[filter]) + 1
            Repeat_time = np.round(Repeat_time, 0)
            for i in range(len(Repeat_time)):
                repeat_time = int(Repeat_time[i])
                oversampled_labels.extend(
                    repeat(oversample_events_index[i], repeat_time)
                )
                oversampled_picks.extend(repeat(oversample_p_pick[i], repeat_time))
            oversampled_labels = np.array(oversampled_labels)
            oversampled_picks = np.array(oversampled_picks)
            ok_events_index = np.concatenate(
                (ok_events_index, oversampled_labels), axis=0
            )
            p_picks = np.concatenate((p_picks, oversampled_picks), axis=0)
        if weight_label:
            labels = labels.flatten()
            classifier = intensity_classifier(label=label_key)
            output_array = classifier.classify(labels)
            label_class, counts = np.unique(output_array, return_counts=True)
            label_counts = {}
            for i, label in enumerate(label_class):
                label_counts[int(label)] = counts[i]
            samples_weight = np.array([1 / label_counts[int(i)] for i in output_array])

        Events_index = []
        Weight = []
        for I, event in enumerate(ok_event_id):
            single_event_index = ok_events_index[
                np.where(ok_events_index[:, 0] == event)[0]
            ]
            single_event_p_picks = p_picks[np.where(ok_events_index[:, 0] == event)[0]]
            # if event==24784:
            #     single_event_index=np.delete(single_event_index, [295], 0)
            #     single_event_p_picks=np.delete(single_event_p_picks, [295], 0)
            if weight_label:
                single_event_label_weight = samples_weight[
                    np.where(ok_events_index[:, 0] == event)[0]
                ]
            if sort_by_picks:
                sort = single_event_p_picks.argsort()
                single_event_p_picks = single_event_p_picks[sort]
                single_event_index = single_event_index[sort]
                if weight_label:
                    single_event_label_weight = single_event_label_weight[sort]
            if len(single_event_index) > max_station_num:
                time = int(np.ceil(len(single_event_index) / 25))  # 無條件進位
                # np.array_split(single_event_index, 25)
                splited_index = np.array_split(
                    single_event_index,
                    np.arange(max_station_num, max_station_num * time, max_station_num),
                )
                if weight_label:
                    splited_weight = np.array_split(
                        single_event_label_weight,
                        np.arange(
                            max_station_num, max_station_num * time, max_station_num
                        ),
                    )
                for i in range(time):
                    Events_index.append([splited_index[0], splited_index[i]])
                    if weight_label:
                        Weight.append(np.mean(splited_weight[i]))

            else:
                Events_index.append([single_event_index, single_event_index])
                if weight_label:
                    Weight.append(np.mean(single_event_label_weight))

        # specific_index=Events_index[400]
        self.data_path = data_path
        self.mode = mode
        self.event_metadata = event_metadata
        self.trace_metadata = trace_metadata
        self.input_type = input_type
        self.label = label_key
        self.labels = labels
        self.ok_events_index = ok_events_index
        self.ok_event_id = ok_event_id
        if weight_label:
            self.weight = Weight
        self.sampling_rate = sampling_rate
        self.data_length_sec = data_length_sec
        self.metadata = metadata
        self.events_index = Events_index
        self.p_picks = p_picks
        self.oversample = oversample
        self.max_station_num = max_station_num
        self.label_target = label_target
        self.mask_waveform_sec = mask_waveform_sec
        self.mask_waveform_random = mask_waveform_random

    def __len__(self):
        return len(self.events_index)

    def __getitem__(self, index):
        specific_index = self.events_index[index]
        with h5py.File(self.data_path, "r") as f:
            # for index in specific_index: #event loop
            specific_waveforms = []
            stations_location = []
            label_targets_location = []
            labels = []
            seen_P_picks = []
            labels_time = []
            P_picks = []
            for eventID in specific_index[0]:  # trace waveform
                waveform = f["data"][str(eventID[0])][f"{self.input_type}_traces"][
                    eventID[1]
                ][: (self.data_length_sec * self.sampling_rate), :]
                station_location = f["data"][str(eventID[0])]["station_location"][
                    eventID[1]
                ]
                waveform = np.pad(
                    waveform,
                    (
                        (0, self.data_length_sec * self.sampling_rate - len(waveform)),
                        (0, 0),
                    ),
                    "constant",
                )
                p_pick = f["data"][str(eventID[0])]["p_picks"][eventID[1]]
                specific_waveforms.append(waveform)
                # print(f"first {waveform.shape}")
                stations_location.append(station_location)
                seen_P_picks.append(p_pick)
            for eventID in specific_index[1]:  # target postion & pga
                station_location = f["data"][str(eventID[0])]["station_location"][
                    eventID[1]
                ]
                label = np.array(
                    f["data"][str(eventID[0])][f"{self.label}"][eventID[1]]
                ).reshape(1, 1)
                p_pick = f["data"][str(eventID[0])]["p_picks"][eventID[1]]
                label_time = f["data"][str(eventID[0])][f"{self.label}_time"][
                    eventID[1]
                ]
                label_targets_location.append(station_location)
                labels.append(label)
                P_picks.append(p_pick)
                labels_time.append(label_time)
            if (
                len(stations_location) < self.max_station_num
            ):  # triggered station < max_station_number (default:25) zero padding
                for zero_pad_num in range(
                    self.max_station_num - len(stations_location)
                ):
                    # print(f"second {waveform.shape}")
                    specific_waveforms.append(np.zeros_like(waveform))
                    stations_location.append(np.zeros_like(station_location))
            # print("================")
            if (
                len(label_targets_location) < self.label_target
            ):  # triggered station < pga_target_number (default:15) zero padding
                for zero_pad_num in range(
                    self.label_target - len(label_targets_location)
                ):
                    label_targets_location.append(np.zeros_like(station_location))
                    labels.append(np.zeros_like(label))
            Specific_waveforms = np.array(specific_waveforms)
            if self.mask_waveform_random:
                random_mask_sec = np.random.randint(self.mask_waveform_sec, 10)
                Specific_waveforms[
                    :, seen_P_picks[0] + (random_mask_sec * self.sampling_rate) :, :
                ] = 0
                for i in range(len(seen_P_picks)):
                    if seen_P_picks[i] > seen_P_picks[0] + (
                        random_mask_sec * self.sampling_rate
                    ):
                        Specific_waveforms[i, :, :] = 0
            elif self.mask_waveform_sec:
                Specific_waveforms[
                    :,
                    seen_P_picks[0] + (self.mask_waveform_sec * self.sampling_rate) :,
                    :,
                ] = 0
                for i in range(len(seen_P_picks)):
                    if seen_P_picks[i] > seen_P_picks[0] + (
                        self.mask_waveform_sec * self.sampling_rate
                    ):
                        Specific_waveforms[i, :, :] = 0
            Stations_location = np.array(stations_location)
            label_targets_location = np.array(label_targets_location)
            labels = np.array(labels)
            P_picks = np.array(P_picks)
            labels_time = np.array(labels_time)
        if self.mode == "train":
            outputs = {
                "waveform": Specific_waveforms,
                "sta": Stations_location,
                "target": label_targets_location,
                "label": labels,
            }
            return outputs
        else:
            P_picks = np.array(P_picks)
            labels_time = np.array(labels_time)
            outputs = {
                "waveform": Specific_waveforms,
                "sta": Stations_location,
                "target": label_targets_location,
                "label": labels,
                "EQ_ID": specific_index[0],
                "p_picks": P_picks,
                f"{self.label}_time": labels_time,
            }
            # others_info = {
            #     "EQ_ID": specific_index[0],
            #     "p_picks": P_picks,
            #     "pga_time": labels_time,
            # }
            return outputs


class multiple_station_dataset_outputs(Dataset):
    def __init__(
        self,
        data_path,
        sampling_rate=200,
        data_length_sec=30,
        test_year=2018,
        mode="train",
        limit=None,
        oversample=1,
        oversample_mag=4,
        input_type=["acc", "vel", "dis"],
        label_keys=["pga", "pgv"],
        mask_waveform_sec=None,
        mask_waveform_random=False,
        max_station_num=25,
        label_target=25,
        sort_by_picks=True,
        weight_label=False,
    ):
        event_metadata = pd.read_hdf(data_path, "metadata/event_metadata")
        trace_metadata = pd.read_hdf(data_path, "metadata/traces_metadata")
        if mode == "train":
            event_test_mask = [
                int(year) != test_year for year in event_metadata["year"]
            ]
            trace_test_mask = [
                int(year) != test_year for year in trace_metadata["year"]
            ]
            event_metadata = event_metadata[event_test_mask]
            trace_metadata = trace_metadata[trace_test_mask]
        elif mode == "test":
            event_test_mask = [
                int(year) == test_year for year in event_metadata["year"]
            ]
            trace_test_mask = [
                int(year) == test_year for year in trace_metadata["year"]
            ]
            event_metadata = event_metadata[event_test_mask]
            trace_metadata = trace_metadata[trace_test_mask]

        if limit:
            event_metadata = event_metadata.iloc[:limit]
        metadata = {}
        data = {}
        with h5py.File(data_path, "r") as f:
            decimate = 1

            skipped = 0
            contained = []
            events_index = np.zeros((1, 2), dtype=int)
            # events_index=[]
            for _, event in event_metadata.iterrows():
                event_name = str(int(event["EQ_ID"]))
                # print(event_name)
                if (
                    event_name not in f["data"]
                ):  # use catalog eventID campare to data eventID
                    skipped += 1
                    contained += [False]
                    continue
                contained += [True]
                g_event = f["data"][event_name]
                for key in g_event:
                    if key not in data:
                        data[key] = []
                    if key == f"{input_type[0]}_traces":
                        index = np.arange(g_event[key].shape[0]).reshape(-1, 1)
                        eventid = (
                            np.array([str(event_name)] * g_event[key].shape[0])
                            .astype(np.int32)
                            .reshape(-1, 1)
                        )
                        single_event_index = np.concatenate([eventid, index], axis=1)
                    if key == label_keys[0]:
                        data[key] += [g_event[key][()]]
                    if key == label_keys[1]:
                        data[key] += [g_event[key][()]]
                    if key == "p_picks":
                        data[key] += [g_event[key][()]]
                    if key == "station_name":
                        data[key] += [g_event[key][()]]
                    if key == "p_picks":
                        data[key][-1] //= decimate

                events_index = np.append(events_index, single_event_index, axis=0)
                # events_index.append(single_event_index)
            events_index = np.delete(events_index, [0], 0)
        stations = np.concatenate(data["station_name"], axis=0)
        picks = np.concatenate(data["p_picks"], axis=0)
        mask = (events_index != 0).any(axis=1)
        ok_events_index = events_index[mask]
        ok_event_id = np.intersect1d(
            np.array(event_metadata["EQ_ID"].values), ok_events_index
        )
        stations = np.expand_dims(np.expand_dims(stations, axis=1), axis=2)
        p_picks = picks[mask]
        stations = stations[mask]
        if oversample > 1:
            oversampled_catalog = []
            filter = event_metadata["magnitude"] >= oversample_mag
            oversample_catalog = np.intersect1d(
                np.array(event_metadata[filter]["EQ_ID"].values), ok_events_index
            )
            for eventid in oversample_catalog:
                catch_mag = event_metadata["EQ_ID"] == eventid
                mag = event_metadata[catch_mag]["magnitude"]
                repeat_time = int(oversample ** (mag - 1) - 1)
                oversampled_catalog.extend(repeat(eventid, repeat_time))

            oversampled_catalog = np.array(oversampled_catalog)
            ok_event_id = np.concatenate([ok_event_id, oversampled_catalog])
        samples_weight = []
        for label_key in label_keys:
            labels = np.concatenate(data[label_key], axis=0)
            # label is nan mask
            # mask = np.logical_and(mask, ~np.isnan(labels))

            labels = np.expand_dims(np.expand_dims(labels, axis=1), axis=2)
            labels = labels[mask]

            if weight_label:
                labels = labels.flatten()
                classifier = intensity_classifier(label=label_key)
                output_array = classifier.classify(labels)
                label_class, counts = np.unique(output_array, return_counts=True)
                label_counts = {}
                for i, label in enumerate(label_class):
                    label_counts[int(label)] = counts[i]
                samples_weight.append(
                    np.array([1 / label_counts[int(i)] for i in output_array])
                )
        if weight_label:
            samples_weight_mean = (samples_weight[0] + samples_weight[1]) / 2

        Events_index = []
        Weight = []
        for I, event in enumerate(ok_event_id):
            single_event_index = ok_events_index[
                np.where(ok_events_index[:, 0] == event)[0]
            ]
            single_event_p_picks = p_picks[np.where(ok_events_index[:, 0] == event)[0]]
            if weight_label:
                single_event_label_weight = samples_weight_mean[
                    np.where(ok_events_index[:, 0] == event)[0]
                ]
            if sort_by_picks:
                sort = single_event_p_picks.argsort()
                single_event_p_picks = single_event_p_picks[sort]
                single_event_index = single_event_index[sort]
                if weight_label:
                    single_event_label_weight = single_event_label_weight[sort]
            if len(single_event_index) > max_station_num:
                time = int(np.ceil(len(single_event_index) / 25))  # 無條件進位
                # np.array_split(single_event_index, 25)
                splited_index = np.array_split(
                    single_event_index,
                    np.arange(max_station_num, max_station_num * time, max_station_num),
                )
                if weight_label:
                    splited_weight = np.array_split(
                        single_event_label_weight,
                        np.arange(
                            max_station_num, max_station_num * time, max_station_num
                        ),
                    )
                for i in range(time):
                    Events_index.append([splited_index[0], splited_index[i]])
                    if weight_label:
                        Weight.append(np.mean(splited_weight[i]))

            else:
                Events_index.append([single_event_index, single_event_index])
                if weight_label:
                    Weight.append(np.mean(single_event_label_weight))

        # specific_index=Events_index[400]
        self.data_path = data_path
        self.mode = mode
        self.event_metadata = event_metadata
        self.trace_metadata = trace_metadata
        self.input_type = input_type
        self.label_keys = label_keys
        if weight_label:
            self.weight = Weight
        self.sampling_rate = sampling_rate
        self.data_length_sec = data_length_sec
        self.metadata = metadata
        self.events_index = Events_index
        self.max_station_num = max_station_num
        self.label_target = label_target
        self.mask_waveform_sec = mask_waveform_sec
        self.mask_waveform_random = mask_waveform_random

    def __len__(self):
        return len(self.events_index)

    def __getitem__(self, index):
        specific_index = self.events_index[index]
        with h5py.File(self.data_path, "r") as f:
            # for index in specific_index: #event loop
            specific_waveforms = {"acc": [], "vel": [], "dis": []}
            stations_location = []
            label_targets_location = []
            labels = {"pga": [], "pgv": []}
            seen_P_picks = []
            labels_time = {"pga_time": [], "pgv_time": []}
            P_picks = []
            for eventID in specific_index[0]:  # trace waveform
                for key in self.input_type:
                    waveform = f["data"][str(eventID[0])][f"{key}_traces"][eventID[1]][
                        : self.data_length_sec * self.sampling_rate
                    ]
                    waveform = np.pad(
                        waveform,
                        (
                            (
                                0,
                                self.data_length_sec * self.sampling_rate
                                - len(waveform),
                            ),
                            (0, 0),
                        ),
                        "constant",
                    )
                    specific_waveforms[key].append(waveform)
                p_pick = f["data"][str(eventID[0])]["p_picks"][eventID[1]]
                station_location = f["data"][str(eventID[0])]["station_location"][
                    eventID[1]
                ]
                # print(f"first {waveform.shape}")
                stations_location.append(station_location)
                seen_P_picks.append(p_pick)
            for eventID in specific_index[1]:  # target postion & pga
                station_location = f["data"][str(eventID[0])]["station_location"][
                    eventID[1]
                ]
                p_pick = f["data"][str(eventID[0])]["p_picks"][eventID[1]]
                for key in self.label_keys:
                    label = np.array(
                        f["data"][str(eventID[0])][key][eventID[1]]
                    ).reshape(1, 1)
                    label_time = f["data"][str(eventID[0])][f"{key}_time"][eventID[1]]
                    labels[key].append(label)
                    labels_time[f"{key}_time"].append(label_time)
                label_targets_location.append(station_location)
                P_picks.append(p_pick)
            if (
                len(stations_location) < self.max_station_num
            ):  # triggered station < max_station_number (default:25) zero padding
                for zero_pad_num in range(
                    self.max_station_num - len(stations_location)
                ):
                    stations_location.append(np.zeros_like(station_location))
                    for key in self.input_type:
                        specific_waveforms[key].append(np.zeros_like(waveform))
            if (
                len(label_targets_location) < self.label_target
            ):  # triggered station < pga_target_number (default:15) zero padding
                for zero_pad_num in range(
                    self.label_target - len(label_targets_location)
                ):
                    label_targets_location.append(np.zeros_like(station_location))
                    for key in self.label_keys:
                        labels[key].append(np.zeros_like(label))
            outputs_waveform = {}
            for key in self.input_type:
                Specific_waveforms = np.array(specific_waveforms[key])
                if self.mask_waveform_random:
                    random_mask_sec = np.random.randint(self.mask_waveform_sec, 10)
                    Specific_waveforms[
                        :, seen_P_picks[0] + (random_mask_sec * self.sampling_rate) :, :
                    ] = 0
                    for i in range(len(seen_P_picks)):
                        if seen_P_picks[i] > seen_P_picks[0] + (
                            random_mask_sec * self.sampling_rate
                        ):
                            Specific_waveforms[i, :, :] = 0
                elif self.mask_waveform_sec:
                    Specific_waveforms[
                        :,
                        seen_P_picks[0]
                        + (self.mask_waveform_sec * self.sampling_rate) :,
                        :,
                    ] = 0
                    for i in range(len(seen_P_picks)):
                        if seen_P_picks[i] > seen_P_picks[0] + (
                            self.mask_waveform_sec * self.sampling_rate
                        ):
                            Specific_waveforms[i, :, :] = 0
                outputs_waveform[key] = Specific_waveforms

            Stations_location = np.array(stations_location)
            label_targets_location = np.array(label_targets_location)
            P_picks = np.array(P_picks)
            labels_time[f"{self.label_keys[0]}_time"] = np.array(
                labels_time[f"{self.label_keys[0]}_time"]
            )
            labels_time[f"{self.label_keys[0]}_time"] = np.array(
                labels_time[f"{self.label_keys[1]}_time"]
            )
        if self.mode == "train":
            outputs = {
                "acc": outputs_waveform["acc"],
                "vel": outputs_waveform["vel"],
                "dis": outputs_waveform["dis"],
                "sta": Stations_location,
                "target": label_targets_location,
                f"{self.label_keys[0]}": np.array(labels[f"{self.label_keys[0]}"]),
                f"{self.label_keys[1]}": np.array(labels[f"{self.label_keys[1]}"]),
            }
            return outputs
        else:
            P_picks = np.array(P_picks)
            labels_time = np.array(labels_time)
            outputs = {
                "acc": outputs_waveform["acc"],
                "vel": outputs_waveform["vel"],
                "dis": outputs_waveform["dis"],
                "sta": Stations_location,
                "target": label_targets_location,
                f"{self.label_keys[0]}": np.array(labels[f"{self.label_keys[0]}"]),
                f"{self.label_keys[1]}": np.array(labels[f"{self.label_keys[1]}"]),
                "EQ_ID": specific_index[0],
                "p_picks": P_picks,
                f"{self.label_keys[0]}_time": labels_time[f"{self.label_keys[0]}_time"],
                f"{self.label_keys[1]}_time": labels_time[f"{self.label_keys[1]}_time"],
            }
            # others_info = {
            #     "EQ_ID": specific_index[0],
            #     "p_picks": P_picks,
            #     "pga_time": labels_time,
            # }
            return outputs


class CustomSubset(Subset):
    """A custom subset class"""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices.tolist())
        self.weight = np.array(dataset.weight)[self.indices]  # 保留weight属性

    def __getitem__(self, idx):
        output = self.dataset[self.indices[idx]]
        return output

    def __len__(self):
        return len(self.indices)


# origin_data = multiple_station_dataset_new(
#     "D:/TEAM_TSMIP/data/TSMIP_filtered.hdf5",
#     mode="train",
#     mask_waveform_sec=3,
#     oversample=1,
#     oversample_mag=4,
#     input_type="acc",
#     label_key="pgv",
#     weight_label=True,
# )
# train_set_size = int(len(origin_data) * 0.8)
# valid_set_size = len(origin_data) - train_set_size
# indice=np.arange(len(origin_data))
# np.random.seed(0)
# np.random.shuffle(indice)
# train_indice,test_indice=np.array_split(indice,[train_set_size])
# train_dataset=CustomSubset(origin_data,train_indice)
# val_dataset=CustomSubset(origin_data,test_indice)
# from torch.utils.data import WeightedRandomSampler

# train_sampler=WeightedRandomSampler(weights=train_dataset.weight,num_samples=len(train_dataset),replacement=True)
# train_loader=DataLoader(dataset=train_dataset,batch_size=16,
#                                 sampler=train_sampler,shuffle=False,drop_last=True)

# batch_size=16
# loader=DataLoader(dataset=origin_data,batch_size=batch_size,shuffle=True)
# a=0
# for sample in origin_data[850]:
# # import matplotlib.pyplot as plt
#     fig,ax=plt.subplots(13,1,figsize=(14,14))
#     for i in range(13):
#         ax[i].plot(origin_data[15276][0][i,:,: ])
#         fig.savefig(f"data/dataset output/{15276}_vel.png")
#         plt.close()
#     a+=1
#     if a>150:
#         break

# full_data = multiple_station_dataset_new(
#     "D:/TEAM_TSMIP/data/TSMIP_filtered.hdf5",
#     mode="train",
#     mask_waveform_sec=3,
#     weight_label=False,
#     oversample=1,
#     test_year=2016,
#     mask_waveform_random=True,
#     label_key="pga",
#     input_type="acc",
#     data_length_sec=10
# )

# train_loader=DataLoader(dataset=full_data,batch_size=16,drop_last=True)

# from tqdm import tqdm
# for sample in tqdm(train_loader):
#     if sample["waveform"].shape[2]!=2000:
#         print(sample["waveform"].shape)
#         break
