import pytest
from VASA.VASA.reduce_vasa_df import *


class TestVASAReucing:
    @pytest.mark.parametrize("a", [(1)])
    def test_reduce_by_count_hh(self, a):
        assert a == 1
