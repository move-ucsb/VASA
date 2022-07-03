import pytest
import numpy as np

from VASA.VASA.missing_value_handling import moving_average, combine_ma


class TestMissingValues:
    @pytest.mark.parametrize("inputs", [(np.array([1, 2, 3, 4, 5, 6, 7]))])
    def test_moving_average_length(self, inputs):
        assert len(moving_average(inputs)) == len(inputs)

    @pytest.mark.parametrize("inputs", [(np.full(7, np.NaN))])
    def test_moving_average_nan(self, inputs):
        ma = moving_average(inputs)
        assert len(ma) == len(inputs)
        for i in ma:
            assert i == 0

    @pytest.mark.parametrize("inputs", [(np.array([1, 1, 1, np.NaN, 1, 1, 1]))])
    def test_moving_average_missing_average(self, inputs):
        # missing value should be 1, harder to test edges
        assert moving_average(inputs)[3] == 1

    @pytest.mark.parametrize("inputs", [(np.array([1, 2, 3, 4, 5, 6, 7]))])
    def test_combine_ma_no_missing(self, inputs):
        assert np.all(combine_ma(inputs) == inputs)

    @pytest.mark.parametrize("inputs", [(np.array([1, 1, 1, np.NaN, 1, 1, 1]))])
    def test_combine_ma_missing_values(self, inputs):
        assert np.all(combine_ma(inputs) == np.ones(7))
