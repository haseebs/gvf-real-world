import torch
import numpy as np
from torch.utils.data import Dataset
from statistics import mean, variance

class TDDataset(Dataset):
    def __init__(self, dataset_file, included_files, cummulants, returns, truncated_returns=None, starting_idx=None, ending_idx=None):
        self.dataset = dataset_file
        self.cummulants = cummulants
        self.returns = returns
        self.truncated_returns = truncated_returns

        # get the starting and the ending row indices for the current split
        self.starting_idx = self.dataset['file_indices'][included_files[0]][0]
        # we do -1 so that we can sample s_t+1
        self.ending_idx = self.dataset['file_indices'][included_files[-1]][1] - 1
        
        # when we need to provide the split ranges manually
        if starting_idx:
            self.starting_idx = starting_idx
        if ending_idx:
            self.ending_idx = ending_idx

        print(self.starting_idx, self.ending_idx)

        self.return_variance = variance(self.returns[self.starting_idx:self.ending_idx])

    def __len__(self):
        return self.ending_idx - self.starting_idx

    def __getitem__(self, idx):
        transformed_idx = idx + self.starting_idx
        # we need this error checking but it is too slow...
        #if (np.any(transformed_idx > self.ending_idx) or np.any(transformed_idx < self.starting_idx)):
        #    raise ValueError("invalid idx for the current split")

        state = self.dataset['features'][transformed_idx]
        next_state = self.dataset['features'][transformed_idx+1]
        returns = self.returns[transformed_idx]
        cummulant = self.cummulants[transformed_idx]
        #truncated_returns = self.truncated_returns[idx] # these are computed on the non-transformed idx
        return state, next_state, returns, cummulant


class ReplayBuffer(TDDataset):
    # This replay buffer should be initialized with training set when doing validation 
    # and the validation test when doing testing
    def __init__(self, dataset_file, included_prev_files, cummulants, returns, included_curr_files, buffer_size):
        TDDataset.__init__(self, dataset_file, included_prev_files, cummulants, returns, None)
        self.return_variance = None
        self.buffer_size = buffer_size
        self.starting_idx = self.ending_idx - self.buffer_size 
        self.ending_idx_current_set = self.dataset['file_indices'][included_curr_files[-1]][1] - 1

    def increment_buffer(self, n, current_index):
        assert self.ending_idx + n < current_index # make sure that future obs are not leaked into buffer
        while self.ending_idx + 1 <  self.ending_idx_current_set and n: # make sure we dont go out of bounds
            n -= 1
            self.starting_idx += 1
            self.ending_idx += 1    

    def sample(self, num_samples):
        random_indices = np.random.choice(range(self.starting_idx, self.ending_idx), num_samples, replace=False)
        random_indices.sort() # hdf5 requires indices in increasing order
        random_indices -= 1 # so that we always have a valid next_state to sample
        state = torch.FloatTensor(self.dataset['features'][random_indices])
        next_state = torch.FloatTensor(self.dataset['features'][random_indices+1])
        returns = torch.FloatTensor(self.returns[random_indices])
        cummulant = torch.FloatTensor(self.cummulants[random_indices])
        return state, next_state, returns, None, cummulant


class LookbackDatasetIndependent(TDDataset):
    def __init__(self, dataset_file, included_files, cummulants, returns, lookback_window_size=100, skip_lookback_steps=False, prediction_horizon=100, starting_idx=None, ending_idx=None):
        """
        Use this class only with the LinearD baseline. This class exists separately purely for speedup reasons, the state contains
        only the relevant sensor than the state of the entire system. This is how the time-series folks do it.

        skip_lookback_steps: we skip <lookback_window_size> steps from the beginning if this is True.
            We want to do this during the training set in order to prevent out of bound leakage.
            Can be set to False during validation/test since we assume continuity in our dataset,
            so the lookback_window samples are sure to exist anyway.

        starting_idx, ending_idx: ignore the indices from file separations, and use these if not None. Need to refactor to combine it with baseclass vars
        """
        returns = np.asarray(returns.tolist()) # to fix this bs bug....
        TDDataset.__init__(self, dataset_file, included_files, cummulants, returns, None)
        if starting_idx:
            self.starting_idx = starting_idx
        if ending_idx:
            self.ending_idx = ending_idx
        self.lookback_window_size = lookback_window_size
        if skip_lookback_steps:
            self.starting_idx += self.lookback_window_size
        self.return_variance = variance(self.returns[self.starting_idx:self.ending_idx])
        self.ending_idx -= prediction_horizon # we dont want leakage of returns from out of bounds. These returns are n-step returns

        # optimization: converting the entire thing to tensors for at once for speedup
        self.cummulants = torch.FloatTensor(self.cummulants)
        self.returns = torch.FloatTensor(self.returns).float()

    def __getitem__(self, idx):
        transformed_idx = idx + self.starting_idx
        # we'll operate directly on cummulants here to make it faster (yes it is significantly faster)
        #TODO edit this for speedup
        state = self.cummulants[transformed_idx - self.lookback_window_size : transformed_idx + 1] #+1 because last idx is not included in slicing
        #state = self.dataset['features'][transformed_idx - self.lookback_window_size : transformed_idx + 1] #+1 because last idx is not included in slicing
        returns = self.returns[transformed_idx]
        cummulant = self.cummulants[transformed_idx]
        return state, state, returns, cummulant



class LookbackDataset(TDDataset):
    def __init__(self, dataset_file, included_files, cummulants, returns, lookback_window_size=100, skip_lookback_steps=False, prediction_horizon=100):
        """
        The class returns the entire state of the system in the "state" variable
        skip_lookback_steps: we skip <lookback_window_size> steps from the beginning if this is True.
            We want to do this during the training set in order to prevent out of bound leakage.
            Can be set to False during validation/test since we assume continuity in our dataset,
            so the lookback_window samples are sure to exist anyway.
        """
        returns = np.asarray(returns.tolist()) # to fix this bs bug....
        TDDataset.__init__(self, dataset_file, included_files, cummulants, returns, None)
        self.lookback_window_size = lookback_window_size
        if (skip_lookback_steps):
            self.starting_idx += self.lookback_window_size
        self.return_variance = variance(self.returns[self.starting_idx:self.ending_idx])
        self.ending_idx -= prediction_horizon # we dont want leakage of returns from out of bounds. These returns are n-step returns

    def __getitem__(self, idx):
        transformed_idx = idx + self.starting_idx
        state = self.dataset['features'][transformed_idx - self.lookback_window_size : transformed_idx + 1] #+1 because last idx is not included in slicing
        #next_state = self.dataset['features'][transformed_idx + 1 - self.lookback_window_size : transformed_idx + 2]
        returns = self.returns[transformed_idx]
        cummulant = self.cummulants[transformed_idx]
        return state, _, returns, cummulant