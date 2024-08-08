# GVFs in the Real World: Making Predictions Online for Water Treatment

**Paper:** [GVFs in the real world: making predictions online for water treatment](https://link.springer.com/article/10.1007/s10994-023-06413-x).

**Data exploration collab:** Coming soon :)


## Setup

- Resolve dependencies by executing the following command:
```bash
pip install -r requirements.txt
```

## Dataset
Dataset used for experiments contains only the months of November 2022 and May 2023. You can get these [here](https://drive.google.com/file/d/1h0C8bfmbpgoHns24KZn3_AFAP7Ihl0PM/view?usp=share_link). We have additionally released the data for an entire year [here](https://drive.google.com/file/d/1eIOktHZARAhNlwdOrGqYZJGDLrtcrtsz/view?usp=sharing).

![overview](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs10994-023-06413-x/MediaObjects/10994_2023_6413_Fig5_HTML.png?as=webp)

### Preprocessing the data
In this codebase, all the preprocessing and state-construction is done only once. Follow the steps below:
- Unzip the log files into
`data-prep/data-logs`.
- Generate mode durations using
  `notebooks/generate_true_modes_json.ipynb`
- Generate normalization ranges using `data-prep/compute_ranges.py`
- Finally, construct state and save the resulting dataset using
  `data-prep/prepare_data_from_logs.py`

## Running the code
The results are stored in a database. Make sure that your credentials
are stored in `credentials.json`. Afterwards, you can run the code as
follows:
```bash
python train_onlinetd.py --name "experiment_name_for_database" --sensor "PIT300"
--training_files "0-4" --val_files "5" --testing_files "6-7"
```
The file ranges in the arguments above refer to the file index in the
`data-logs` folder.

### Sensor tags
Refer to the following table for the tags of sensors used in the paper:
| Tag    | Description            |
|--------|------------------------|
| ZW_PIT300.DATA | Membrane Pressure      |
| ZW_TIT101.DATA | Influent Temperature   |
| ZW_PIT101.DATA | Inlet Pressure         |
| ZW_FIT101.DATA | Inlet Flow Transmitter |
| ZW_PIT520.DATA | Drain Reject Pressure  |

## Citation
If you found our work helpful in your research, consider citing the following:
```
@article{janjua2023gvfs,
  title={GVFs in the real world: making predictions online for water treatment},
  author={Janjua, Muhammad Kamran and Shah, Haseeb and White, Martha and Miahi, Erfan and Machado, Marlos C and White, Adam},
  journal={Machine Learning},
  pages={1--31},
  year={2023},
  publisher={Springer}
}
```
