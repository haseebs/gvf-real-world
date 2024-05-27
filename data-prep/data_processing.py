import copy
import json
from typing import Dict, List
from features import (
    Featurizer,
    ModeCountdownEncoding,
    MinMaxNormalizer,
    OnehotEncoder,
    TimestampCyclicalEncoder,
    ExpTraceFeaturizer,
    ModeExtractor,
    BinningFeaturizer,
    LogNameChangeFeaturizer,
    TrueModeLenCountDownEncoding,
    SineCosineModeCountDownEncoding,
    LinearDegradationCountDownEncoding,
    TwoBitEncoder
)


class OnlineCompositeMinimalFeaturizer(Featurizer):
    def __init__(self, ranges, categories):
        self._ranges = ranges
        self._categories = categories

        self._rename = LogNameChangeFeaturizer()
        self._mode_extract = ModeExtractor()
        self._minmax = MinMaxNormalizer(self._ranges)
        self._onehot = OnehotEncoder(self._categories)
        self._cyclical = TimestampCyclicalEncoder()

    def transform(self, x):
        """
        Overall process:
            Input (I) -> [encode time] ->  [one hot] -> [Normalization] -> output (O)
        """
        x = self._rename.transform(x)
        x = self._mode_extract.transform(x)
        x = self._cyclical.transform(x)
        x = self._onehot.transform(x)
        x = self._minmax.transform(x)
        return x

    def transform_array(self, x):
        x_dict = self.transform(x)
        return [x_dict[k] for k in sorted(x_dict.keys())]


class OnlineCompositeFeaturizer:
    def __init__(self, ranges, categories, decay_factor, bins, countdown_len, mode_durations, two_bit_cols):
        self._ranges = ranges
        self._categories = categories
        self._decay_factor = decay_factor
        self._bins = bins
        self._countdown_len = countdown_len
        self._mode_durations = mode_durations
        self._two_bit_cols = two_bit_cols
        # required only with ModeCountdownEncoding (which uses average mode to construct timers)
        # with open("../configs/average_mode_durations.json") as f:
        #     self._average_mode_durations = json.load(
        #         f, object_pairs_hook=lambda x: {int(k): v for k, v in x}
        #     )

        self._rename = LogNameChangeFeaturizer()
        self._mode_extract = ModeExtractor()
        self._minmax = MinMaxNormalizer(self._ranges)
        self._onehot = OnehotEncoder(self._categories)
        self._twobit = TwoBitEncoder(self._two_bit_cols)
        self._cyclical = TimestampCyclicalEncoder()
        # to do linear degradation at the last bit, uncomment this
        # self._count_down = LinearDegradationCountDownEncoding(self._countdown_len,
        #                                                       self._mode_durations)
        # sine/cosine encoding is the default
        self._count_down = SineCosineModeCountDownEncoding(self._countdown_len,
                                                           self._mode_durations)

        # to do the true mode, uncomment this and comment the other count_down
        # self._count_down = TrueModeLenCountDownEncoding(
        #     self._countdown_len, self._mode_durations
        # )

        # to do average mode len timers, uncomment this
        # self._count_down = ModeCountdownEncoding(self._countdown_len, self._average_mode_durations)
        self._exp_trace = ExpTraceFeaturizer(self._decay_factor)
        # do not do binning by default
        # self._binning = BinningFeaturizer(self._bins)

    def transform(self, x, transition_idx):
        """
        Overall process:
            x -> [mode extract] -> [one hot] -> mode
            x -> [encode time] -> [one hot] -> [Normalization] -> normalized sensors
            x -> [count down] -> count-down

            o = [normalized sensors, mode]
            out = [o, binned-trace(o), count-down]
        """

        x = self._rename.transform(x)

        # Extract mode
        x = self._mode_extract.transform(x)

        # mode counting down encoding
        count_down = self._count_down.transform(x, transition_idx)

        # Encode time
        x = self._cyclical.transform(x)

        # One hot
        x = self._onehot.transform(x)

        # Normalization
        x = self._minmax.transform(x)

        # Two Bit
        x = self._twobit.transform(x)

        x_normalized = copy.deepcopy(x)

        # trace
        x = self._exp_trace.transform(x)

        # binning
        # do not do binning, by default.
        # x = self._binning.transform(x)
        return {**x_normalized, **x, **count_down}


class ObservationTraces(Featurizer):
    """
    Do traces here on the observations.
    """

    def __init__(self, decay_factor):
        self._decay_factor = decay_factor
        self._exp_trace = ExpTraceFeaturizer(self._decay_factor)

    def transform(self, x):
        # trace
        x = self._exp_trace.transform(x)
        return x


class ActionTraceBinning(Featurizer):
    """
    Overall process:
        Input (I) -> onehot -> trace -> binning on trace -> output (O)
    """

    def __init__(
            self, categories: List[int], n_bins: List[float], decay_factor: float
    ) -> None:
        self._categories = categories
        self._n_bins = n_bins
        self._decay_factor = decay_factor

        self._onehot = OnehotEncoder({"action": self._categories}, oh_only=True)
        self._exp_trace = ExpTraceFeaturizer(self._decay_factor, traces_only=False)
        self._binning = BinningFeaturizer(
            {f"action${c}_trace": self._n_bins for c in self._categories},
            bins_only=False,
        )

    def transform(self, x: Dict) -> Dict:
        x = self._onehot.transform(x)

        x = self._exp_trace.transform(x)
        x = self._binning.transform(x)
        return x
