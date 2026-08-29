import numpy as np
import pytest

from ghana_cocoa_lulc.change import transition_table


def test_transition_table_counts_area_and_excludes_nodata():
    baseline = np.array([[1, 1], [2, 0]])
    comparison = np.array([[1, 2], [2, 1]])

    result = transition_table(baseline, comparison, pixel_area_ha=0.09)

    assert result["pixel_count"].sum() == 3
    assert result["area_ha"].sum() == pytest.approx(0.27)
    assert set(map(tuple, result[["from_class", "to_class"]].to_numpy())) == {
        (1, 1),
        (1, 2),
        (2, 2),
    }


def test_transition_table_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal shapes"):
        transition_table(np.ones((2, 2)), np.ones((3, 3)), 0.09)

