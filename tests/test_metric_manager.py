import pytest

from matches.loop import IterationType


def test_enum_convertation():
    assert IterationType("auto") == IterationType.AUTO
    assert IterationType("epochs") == IterationType.EPOCHS
    assert IterationType("samples") == IterationType.SAMPLES
    assert IterationType("batches") == IterationType.BATCHES

    with pytest.raises(ValueError):
        IterationType("no-such-type")

    with pytest.raises(ValueError):
        IterationType(1)

    with pytest.raises(ValueError):
        IterationType(10.)
