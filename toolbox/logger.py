import logging
import coloredlogs

log = logging.getLogger('escher')
log.setLevel(logging.INFO)
logging.raiseExceptions = 0

# format='%(asctime)s [%(name)s][%(levelname)s] %(message)s')
coloredlogs.install(logger=log, isatty=True, level='DEBUG',
                    fmt='%(message)s')
