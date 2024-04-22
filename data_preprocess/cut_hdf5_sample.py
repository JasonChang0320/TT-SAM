import h5py
import pandas as pd


data_path="../data/TSMIP_1999_2019_Vs30.hdf5"
init_event_metadata = pd.read_hdf(data_path, "metadata/event_metadata")
trace_metadata = pd.read_hdf(data_path, "metadata/traces_metadata")

sample_eqid=init_event_metadata.query("year==2016")["EQ_ID"]


with h5py.File(data_path, "r") as origin, h5py.File("../data/2016_sample.hdf5", 'w') as sample:
    sample.create_group("data")
    sample.create_group("metadata")

    for eqid in sample_eqid.values:
        print(eqid)
        data = origin["data"][str(eqid)]
        sample_group=sample["data"].create_group(f"{eqid}")

        for col in data:
            attr=data[f"{col}"]

            sample_group.copy(attr,col)

init_event_metadata.to_hdf('2016_sample.hdf5', key="metadata/event_metadata", mode="a", format="table")
trace_metadata.to_hdf('2016_sample.hdf5', key="metadata/traces_metadata", mode="a", format="table")
