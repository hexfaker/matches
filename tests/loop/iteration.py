from matches.loop.iteration import IterationCounter


def test_iteration_counter():
    counter = IterationCounter()

    # Test integer emulation
    assert counter.current_epoch == 0
    assert counter.current_epoch + 1 == 1

    # Test inc
    counter.current_epoch.inc(2)
    assert counter.current_epoch == 2
    counter.current_epoch.inc()
    assert counter.current_epoch == 3

    counter.current_epoch.reset()
