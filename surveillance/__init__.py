import logging
import sys

__version__ = '0.1.0'


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
