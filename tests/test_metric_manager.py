import pytest


from matches.metric_manager import MetricIterationType


def test_enum_convertation():
    assert MetricIterationType("auto") == MetricIterationType.AUTO
    assert MetricIterationType("epochs") == MetricIterationType.EPOCHS
    assert MetricIterationType("samples") == MetricIterationType.SAMPLES
    assert MetricIterationType("batches") == MetricIterationType.BATCHES

    with pytest.raises(ValueError):
        MetricIterationType("no-such-type")

    with pytest.raises(ValueError):
        MetricIterationType(1)

    with pytest.raises(ValueError):
        MetricIterationType(10.)
