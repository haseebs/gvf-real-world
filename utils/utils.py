import torch
import h5py
import random
import numpy as np
from dataset import TDDataset, ReplayBuffer, LookbackDatasetIndependent
from sklearn import metrics


def get_run(arg_dict, run=0):
    combinations = []

    for key in arg_dict.keys():
        if isinstance(arg_dict[key], list):
            combinations.append(len(arg_dict[key]))

    selected_combinations = []
    for base in combinations:
        selected_combinations.append(run % base)
        run = int(run / base)

    counter = 0
    result_dict = {}

    for key in arg_dict.keys():
        result_dict[key] = arg_dict[key]
        if isinstance(arg_dict[key], list):
            result_dict[key] = arg_dict[key][selected_combinations[counter]]
            counter += 1

    return result_dict


def compute_returns(cummulants, gamma):
    g, g_list = 0.0, []
    for c in cummulants[::-1]:
        g = c + gamma * g
        g_list.append(g)
    return np.asarray(list(reversed(g_list)))


def compute_returns_nstep(cummulants, prediction_horizon):
    rolled_cummulants = np.roll(cummulants, -prediction_horizon) # shift all elements to the left
    rolled_cummulants[rolled_cummulants.shape[0] - prediction_horizon:] = 0 #set last elements to 0
    return rolled_cummulants


def compute_truncated_returns(cummulants, gamma, horizon, starting_idx, ending_idx):
    returns = np.zeros(ending_idx - starting_idx)
    for step, idx in enumerate(range(starting_idx, ending_idx)):
        while idx + horizon > ending_idx:
            horizon -= 1
        returns[step] = np.polynomial.polynomial.polyval(gamma, cummulants[idx:idx+horizon])
    return returns


def get_split(data, file_range, cummulants, returns, starting_idx=None, ending_idx=None):
    included_files = None
    if "," in file_range:  # if multiple separate file ranges are present
        datasets = []
        for files in file_range.split(","):
            datasets.append(get_split(data, files, cummulants, returns, starting_idx, ending_idx))
        return datasets
    elif "-" in file_range:
        included_files = range(int(file_range.split('-')[0]), int(file_range.split('-')[1]) + 1)
    else:
        included_files = [int(file_range)]
    return TDDataset(dataset_file=data,
                     included_files=included_files,
                     cummulants=cummulants,
                     returns=returns,
                     starting_idx=starting_idx,
                     ending_idx=ending_idx)


def get_lookback_split(data, file_range, cummulants, returns, lookback_window_size, skip_lookback_steps, prediction_horizon, starting_idx=None, ending_idx=None):
    included_files = None
    if "," in file_range:  # if multiple separate file ranges are present
        datasets = []
        for files in file_range.split(","):
            datasets.append(get_split(data, files, cummulants, returns))
        return datasets
    elif "-" in file_range:
        included_files = range(int(file_range.split('-')[0]), int(file_range.split('-')[1]) + 1)
    else:
        included_files = [int(file_range)]
    return LookbackDatasetIndependent(dataset_file=data,
                                      included_files=included_files,
                                      cummulants=cummulants,
                                      returns=returns,
                                      lookback_window_size=lookback_window_size,
                                      skip_lookback_steps=skip_lookback_steps,
                                      prediction_horizon=prediction_horizon,
                                      starting_idx=starting_idx,
                                      ending_idx=ending_idx)


def get_replay_split(data, file_range_previous, cummulants, returns, file_range_current, buffer_size):
    def get_included_files(file_range):
        if "-" in file_range:
            included_files = range(int(file_range.split('-')[0]), int(file_range.split('-')[1]) + 1)
        else:
            included_files = [int(file_range)]
        return included_files

    assert "," not in file_range_previous  # if multiple separate file ranges are present
    assert "," not in file_range_current

    included_files = get_included_files(file_range_previous)
    included_curr_files = get_included_files(file_range_current)

    return ReplayBuffer(dataset_file=data,
                        included_prev_files=included_files,
                        cummulants=cummulants,
                        returns=returns,
                        included_curr_files=included_curr_files,
                        buffer_size=buffer_size)


def normalized_mse(y_true, y_pred, var):
    return metrics.mean_squared_error(y_true, y_pred) / var


def compute_scores(y_test, y_pred, var):
    nmse = normalized_mse(y_test, y_pred, var)
    return float(nmse)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True) # cba
    torch.backends.cudnn.deterministic = True


def load_entire_hdf5(dct):
    if isinstance(dct, h5py.Dataset):
        return dct[()]
    ret = {}
    for k, v in dct.items():
        ret[k] = load_entire_hdf5(v)
    return ret


if __name__ == '__main__':
    cummulants = list(range(0,12))
    gamma = 0.5
    horizon = 3
    starting_idx = 2
    ending_idx = 10

    returns = compute_truncated_returns(cummulants,
                                        gamma,
                                        horizon,
                                        starting_idx,
                                        ending_idx)
    print(cummulants)
    print(cummulants[starting_idx:ending_idx])
    print(returns)

