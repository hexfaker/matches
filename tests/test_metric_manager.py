import logging
logging.basicConfig()

import pytest

from matches.loop import IterationType
from matches.loop.metric_manager import _log_non_finit


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


def test_log_non_finit(caplog):
    _log_non_finit("foo", float("nan"))
    assert "foo" in caplog.records[-1].getMessage()
