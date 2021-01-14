import logging
import sys
from functools import partial

import coloredlogs
import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, file, level=logging.NOTSET):
        super().__init__(level)
        self.file = file

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, file=self.file)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class StreamToLogger:
    """
         Fake file-like stream object that redirects writes to a logger instance.
         """

    def __init__(self, logger_name, log_level=logging.INFO):
        self.logger = logging.getLogger(logger_name)
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def configure_logging(file=sys.stdout):
    coloredlogs.StandardErrorHandler = partial(TqdmLoggingHandler, file=file)
    coloredlogs.install()
    logging.captureWarnings(True)
    sys.stdout = StreamToLogger("STDOUT")
    sys.stderr = StreamToLogger("STDERR")
