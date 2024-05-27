"""
    This script prepares the hdf5 files from .log files of plant data.
"""
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from glob import glob
from data_processing import OnlineCompositeFeaturizer, ObservationTraces
import h5py
import numpy as np

class ProcessDataLogFiles:
    """
    This class takes in .log files and processes them to generate pkl files sequentially.
    """

    def __init__(self):
        self.parent_path = (
                "data-logs/"
        )
        self.save_dir = "processed/"
        self.DATA_PATHS = [Path(x) for x in glob(self.parent_path.__str__() + "/*.log.gz")]
        self.DATA_PATHS.sort()

        # data pre-processing configuration
        self.bins = [0.14, 0.28, 0.42, 0.56, 0.7, 0.84, 0.98]
        self.trace_decay = 0.99
        self.countdown_len = 7
        
        self.file_idxes = []
        self.data_file = h5py.File(f"{self.save_dir}" + "/processed_data.hdf5", "w")

    def read_csv(self, path):
        df = pd.read_csv(
            path,
            skiprows=[
                1,
            ],
        )
        for c in df.columns:
            try:
                df[c] = df[c].astype(float)
            except:
                pass
        return df


    def read_json(self, path):
        with open(path) as f:
            return json.load(f)

    def load_and_transform(self, path):
        print("read csv files...")
        data = self.read_csv(path)

        data.columns = [
            c.split("Program:")[-1].replace("ZW.", "ZW_") if c != "timestamp" else c
            for c in data.columns
        ]

        print("generating features...")
        range_info = self.read_json("computed_ranges/normalization_ranges.json")  # use updated ranges
        columns = range_info["request_tags"] + ["timestamp"]
        data = data[columns].reset_index(drop=True).fillna(0)

        mode_ranges = self.read_json("computed_ranges/november_modes.json")
        mode_duration_ranges = mode_ranges[path.name]
        mode_duration_ranges = {int(key): value for key, value in mode_duration_ranges.items()}

        featurizer = OnlineCompositeFeaturizer(
            ranges=range_info["numerical"],
            categories=range_info["categorical"],
            decay_factor=self.trace_decay,
            bins=self.bins,
            countdown_len=self.countdown_len,
            mode_durations=mode_duration_ranges,
            two_bit_cols=range_info["2bit_encode"]
        )

        # trace_featurizer = ObservationTraces(decay_factor=self.trace_decay)

        obs_dict = data.to_dict("records")
        data_p = []
        for ix, obs in tqdm(enumerate(obs_dict)):
            obs_p = featurizer.transform(obs, ix)
            data_p.append(obs_p)

        data_p = pd.DataFrame(data_p)
        print(f"Pre-processed data shape: {data_p}")
        return data_p.values.astype(np.float32), data_p.columns.tolist()

    def generate_hdf5(self):
        last_ending_index = 0
        for itr, path in enumerate(self.DATA_PATHS):
            print(f"processing {path.name}...")
            data, labels = self.load_and_transform(path)
            
            # store the starting and ending indices of files
            self.file_idxes.append([last_ending_index, last_ending_index + data.shape[0] - 1])
            last_ending_index += data.shape[0]
            
            if (itr == 0):
                # on first itr, create the datasets
                self.data_file.create_dataset('features', data=data, maxshape=(None, data.shape[1]), chunks=(1024, data.shape[1]))
                self.data_file.create_dataset('labels', data=labels) 
            else:
                # on further itrs, extend the existing dataset
                self.data_file['features'].resize((self.data_file['features'].shape[0] + data.shape[0]), axis=0)
                self.data_file['features'][-data.shape[0]:] = data
        # write the file indices to hdf5 after we're done
        self.data_file.create_dataset('file_indices', data=self.file_idxes)
        self.data_file.close()


if __name__ == "__main__":
    p = ProcessDataLogFiles()
    p.generate_hdf5()
