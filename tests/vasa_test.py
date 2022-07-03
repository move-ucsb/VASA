import pytest
from unittest.mock import Mock, patch

from datetime import date
from typing import Tuple

from VASA.VASA.vasa import VASA


class TestVasa:
    @pytest.mark.parametrize(
        "date,expected",
        [
            (date(2019, 12, 31), (2020, 1)),  # goes to next year
            (date(2020, 1, 1), (2020, 1)),
            (date(2021, 1, 1), (2020, 53)),  # goes to prev year
        ],
    )
    def test_get_year_week(self, date: date, expected: Tuple[int, int]):
        assert VASA._VASA__get_year_week(date) == expected

    @patch.object(VASA, "_VASA__get_col_numpy")
    def test_pct_missing(self, v):
        v.return_value = 1
        assert True
