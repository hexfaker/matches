from matches.callbacks.tensorboard import TensorboardMetricWriterCallback

from matches.loop import Loop


def get_summary_writer(loop: Loop):
    return [
        c.get_sw(loop)
        for c in loop.callbacks
        if isinstance(c, TensorboardMetricWriterCallback)
    ][0]
