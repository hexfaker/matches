import logging

import coloredlogs
import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def configure_logging():
    coloredlogs.StandardErrorHandler = TqdmLoggingHandler
    coloredlogs.install()
    logging.captureWarnings(True)
