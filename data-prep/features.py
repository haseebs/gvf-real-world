import copy
from typing import Dict
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


# exception
class FeatureNotReadyException(Exception):
    pass


# abstract class
class Featurizer(ABC):
    @abstractmethod
    def transform(self, x):
        pass

    def transform_with_idx(self, x, idx):
        pass

    def transform_array(self, x):
        x_dict = self.transform(x)
        return [x_dict[k] for k in sorted(x_dict.keys())]


class LinearDegradationCountDownEncoding:
    def __init__(
            self,
            encoding_len: int,
            mode_durations,
            count_down_only: bool = True,
    ):
        self.encoding_len = encoding_len
        self.mode_durations = mode_durations  # 1: 30 --> mode 1 lasts 30 steps
        self.count_down_only = count_down_only
        self.counter = 1
        self.current_mode = None

    def get_mode_durations(self, transition_idx, current_mode):
        """Gets the total duration of current mode by looking ahead."""
        current_mode_durations = self.mode_durations[current_mode]
        max_len = max([x[1] for x in current_mode_durations])
        for rng in current_mode_durations:
            if rng[0] <= transition_idx < rng[1]:
                duration = rng[2]
            elif transition_idx == max_len:
                duration = rng[2]
        return duration

    def compute_linear_count_down(self, transition_idx, duration):
        """Computes the linear count-down using total duration."""
        return transition_idx/duration

    def transform(self, x, transition_idx):
        new_mode = x["ZW_CTRL.MODE"]
        duration = self.get_mode_durations(transition_idx, new_mode)
        increment = duration // self.encoding_len
        if self.current_mode == new_mode:
            first_one = self.encoding_len - (self.counter // increment) - 1
            self.counter += 1
        else:
            self.counter = 1
            self.current_mode = new_mode
            first_one = self.encoding_len - 1

        if self.count_down_only:
            out = {}
        else:
            out = copy.deepcopy(x)

        for i in range(self.encoding_len):
            if i >= first_one:  # ones
                out[f"mode_enc{i}"] = 1
            elif i < first_one:  # zeros
                out[f"mode_enc{i}"] = 0

        # check if the last bit is set to 1, start count down from there
        if out[f"mode_enc0"] == 1:
            linear_val = self.compute_linear_count_down(self.counter, duration)
            out[f'mode_enc0'] = linear_val
        return out


class SineCosineModeCountDownEncoding:
    def __init__(
            self,
            encoding_len: int,
            mode_durations: Dict[int, int],
    ):
        self.encoding_len = encoding_len
        self.mode_durations = mode_durations
        self.current_mode = None
        self.counter = 1

    def get_mode_durations(self, transition_idx, current_mode):
        """Gets the total duration of current mode by looking ahead (cheating)."""
        current_mode_durations = self.mode_durations[current_mode]
        max_len = max([x[1] for x in current_mode_durations])
        for rng in current_mode_durations:
            if rng[0] <= transition_idx <= rng[1]:
                duration = rng[2]
            elif transition_idx == max_len:
                duration = rng[2]
        return duration

    def get_sin_cosine_transforms(self, transition_idx, duration):
        sine_vector, cosine_vector = [], []
        for ix in range(self.encoding_len):
            multiplier = 2**(ix+1)
            if duration == 0:
                # this mode had no duration
                sine_val = 0
                cosine_val = 0
            else:
                sine_val = np.sin(multiplier*np.pi*(transition_idx/duration))
                cosine_val = np.cos(multiplier*np.pi*(transition_idx/duration))
            sine_vector.append(sine_val)
            cosine_vector.append(cosine_val)
        return sine_vector, cosine_vector

    def transform(self, x, transition_idx):
        new_mode = x["ZW_CTRL.MODE"]
        duration = self.get_mode_durations(transition_idx, new_mode)
        if self.current_mode == new_mode:
            # the mode is still the same
            self.counter += 1
        else:
            self.counter = 1
            self.current_mode = new_mode
        sine_vector, cosine_vector = self.get_sin_cosine_transforms(self.counter, duration)
        out = {}
        for ix, val in enumerate(sine_vector):
            out[f'sine_mode_enc{ix}'] = val
        for ix, val in enumerate(cosine_vector):
            out[f'cos_mode_enc{ix}'] = val
        return out


class TrueModeLenCountDownEncoding:
    def __init__(
            self,
            encoding_len: int,
            mode_durations,
            count_down_only: bool = True,
    ):
        self.encoding_len = encoding_len
        self.mode_durations = mode_durations  # 1: 30 --> mode 1 lasts 30 steps
        self.count_down_only = count_down_only
        self.counter = 1
        self.current_mode = None

    def get_mode_durations(self, transition_idx, current_mode):
        """Gets the total duration of current mode by looking ahead (cheating)."""
        current_mode_durations = self.mode_durations[current_mode]
        max_len = max([x[1] for x in current_mode_durations])
        for rng in current_mode_durations:
            if rng[0] <= transition_idx < rng[1]:
                duration = rng[2]
            elif transition_idx == max_len:
                duration = rng[2]
        return duration

    def transform(self, x, transition_idx):
        new_mode = x["ZW_CTRL.MODE"]
        duration = self.get_mode_durations(transition_idx, new_mode)
        increment = duration // self.encoding_len
        if self.current_mode == new_mode:
            first_one = self.encoding_len - (self.counter // increment) - 1
            self.counter += 1
        else:
            self.counter = 1
            self.current_mode = new_mode
            first_one = self.encoding_len - 1

        if self.count_down_only:
            out = {}
        else:
            out = copy.deepcopy(x)

        for i in range(self.encoding_len):
            if i >= first_one:  # ones
                out[f"mode_enc{i}"] = 1
            elif i < first_one:  # zeros
                out[f"mode_enc{i}"] = 0
        return out


class ModeCountdownEncoding(Featurizer):
    def __init__(
            self,
            encoding_len: int,
            mode_durations: Dict[int, int],
            count_down_only: bool = True,
    ):

        self.encoding_len = encoding_len
        self.mode_durations = mode_durations  # 1: 30 --> mode 1 lasts 30 steps
        self.count_down_only = count_down_only
        self.counter = 1
        self.current_mode = None

    def transform(self, x):
        new_mode = x["ZW_CTRL.MODE"]
        duration = self.mode_durations[new_mode]
        increment = duration // self.encoding_len
        if self.current_mode == new_mode:
            first_one = self.encoding_len - (self.counter // increment) - 1
            self.counter += 1
        else:
            self.counter = 1
            self.current_mode = new_mode
            first_one = self.encoding_len - 1

        if self.count_down_only:
            out = {}
        else:
            out = copy.deepcopy(x)

        for i in range(self.encoding_len):
            if i >= first_one:  # ones
                out[f"mode_enc{i}"] = 1
            elif i < first_one:  # zeros
                out[f"mode_enc{i}"] = 0

        return out


class MinMaxNormalizer(Featurizer):
    def __init__(self, ranges):
        self._ranges = ranges

    def transform(self, x):
        """normalize values into [0, 1]"""
        for col in self._ranges:
            if col not in x:
                continue
            if self._ranges[col][0] == self._ranges[col][1]:
                x[col] = x[col] / (x[col] + 30)
            else:
                x[col] = (x[col] - self._ranges[col][0]) / (
                        self._ranges[col][1] - self._ranges[col][0]
                )
        return x


class TwoBitEncoder(Featurizer):
    """Encode binary values to 2 bits [1,0], [0,1]"""
    def __init__(self, two_bit_cols):
        self._two_bit_cols = two_bit_cols

    def transform(self, x):
        out = copy.deepcopy(x)
        for col in self._two_bit_cols:
            if out[col] == 0.0:
                out[col + "$0"] = 1.0
                out[col + "$1"] = 0.0
            elif out[col] == 1.0:
                out[col + "$0"] = 0.0
                out[col + "$1"] = 1.0
            # delete the original col.
            del out[col]
        return out


class OnehotEncoder(Featurizer):
    def __init__(self, categories, oh_only=True):
        self._categories = categories
        self._oh_only = oh_only

    def transform(self, x):
        out = copy.deepcopy(x)

        for col in self._categories:
            for cat in self._categories[col]:
                out[col + "$" + str(cat)] = int(x[col] == cat)
            if self._oh_only:
                del out[col]

        return out


class TimestampCyclicalEncoder(Featurizer):
    SECONDS_OF_DAY = 60 * 60 * 24

    def transform(self, x):
        if "timestamp" in x and isinstance(x["timestamp"], str):
            ts = x["timestamp"].split(".")[0]
            ts = pd.to_datetime(ts)

            sec_of_day = ts.hour * 60 * 60 + ts.minute * 60 + ts.second

            del x["timestamp"]
            x["ts_sin"] = np.sin(2 * np.pi * sec_of_day / self.SECONDS_OF_DAY)
            x["ts_cos"] = np.cos(2 * np.pi * sec_of_day / self.SECONDS_OF_DAY)
        return x


class StackingFeaturizer(Featurizer):
    def __init__(self, n_stacks):
        self._n_stacks = n_stacks
        self._past_obs = []

    def transform(self, x):
        out = copy.deepcopy(x)
        for i, obs in enumerate(reversed(self._past_obs)):
            for col in obs:
                out[col + "@" + str(i + 1)] = obs[col]

        # fill with zeros for missing values
        i = len(self._past_obs)
        while i < self._n_stacks:
            for col in x:
                out[col + "@" + str(i + 1)] = 0
            i += 1

        self._past_obs.append(x)
        self._past_obs = self._past_obs[-self._n_stacks:]
        return out


class ExpTraceFeaturizer(Featurizer):
    def __init__(self, decay_factor=0.95, traces_only=True):
        self._decay_factor = decay_factor
        self._trace_only = traces_only
        self._e = None

    def transform(self, x):
        if self._e is None:
            self._e = np.zeros(len(x))

        x_vec = np.array(list(x.values()))
        self._e = (1 - self._decay_factor) * x_vec + self._decay_factor * self._e
        self._e[self._e > 1] = 1

        out = copy.deepcopy(x) if not self._trace_only else {}
        for i, k in enumerate(x.keys()):
            out[k + "_trace"] = self._e[i]

        return out


class ModeExtractor(Featurizer):
    def __init__(self):
        self._mode_col = "ZW_CTRL_P.STEP"

    def transform(self, x):
        if self._mode_col in x:
            x["ZW_CTRL.MODE"] = x[self._mode_col] // 100
            x["ZW_CTRL._STEP"] = (x[self._mode_col] % 100) / 100  # normalize to [0, 1]
            del x[self._mode_col]
        return x


class BinningFeaturizer(Featurizer):
    def __init__(self, bins, bins_only=True):
        """
        :param bins: dict or list of bins. if `bins` is a list, all features will use the same
            binning scheme. if `bins` is a dict, only features in the keys of `bins` are binned,
            and each feature can have its own binning scheme.
        """
        self._bins = bins
        self._bins_only = bins_only

    def transform(self, x):
        if self._bins_only:
            out = {}
        else:
            out = copy.deepcopy(x)

        if isinstance(self._bins, dict):
            # only bin features in the keys of `bins`
            for col in self._bins:
                if col not in x:
                    continue
                activated_bin = np.digitize(x[col], self._bins[col])

                for i in range(len(self._bins[col]) + 1):
                    out[f"{col}_bin_{i}"] = int(activated_bin == i)
        else:
            # bin all features in x
            for col in x:
                activated_bin = np.digitize(x[col], self._bins)

                for i in range(len(self._bins) + 1):
                    out[f"{col}_bin_{i}"] = int(activated_bin == i)

        return out


class LogNameChangeFeaturizer(Featurizer):
    """Log data uses a different naming convention than online observations:
    Offline logs: Program:ZW.<TAG>.<type>
    Online observations: ZW_<TAG>.<type>
    """

    def transform(self, x):
        out = {}
        keys = list(x.keys())
        for key in keys:
            if key.startswith("Program"):
                new_key = key.split("Program:")[-1].replace("ZW.", "ZW_")
                out[new_key] = x[key]
            else:
                out[key] = x[key]
        return out


class ObservationSmoothing(Featurizer):
    def __init__(self, decay, means, stds) -> None:
        self._decay = decay
        self._means = means
        self._stds = stds

    def transform(self, x):
        for col in self._means.keys():
            val = x[col]
            mean = self._means[col]
            std = self._stds[col]

            if abs(val - mean) < 2 * std:
                self._means[col] = mean * self._decay + val * (1 - self._decay)

            x[col] = self._means[col]

        return x
