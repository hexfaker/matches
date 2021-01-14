from logging import getLogger
import ignite.utils

# Ignite has ugly logging setup infrastructure
# And this is ugly hack to disable it
ignite.distributed.auto.setup_logger = lambda name, **kwargs: getLogger(name)
