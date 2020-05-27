from fitsnap3.io.input import config
from fitsnap3.parallel_tools import pt
from contextlib import contextmanager
import gzip
from datetime import datetime
import logging


class Output:

    def __init__(self, name):
        self.name = name
        self._screen = config.args.screen
        self._pscreen = config.args.pscreen
        self._nscreen = config.args.nscreen
        self._logfile = config.args.log
        if self._logfile is None:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.DEBUG, filename=self._logfile)
        self.logger = logging.getLogger(__name__)
        pt.set_logger(self.logger)

    def screen(self, *args, **kw):
        if self._pscreen:
            print(*args, flush=True)
        elif self._nscreen:
            pt.sub_print(*args, **kw)
        elif self._screen:
            pt.single_print(*args, **kw)
        else:
            pass

    @pt.rank_zero
    def info(self, msg):
        self.logger.info(msg)

    @pt.rank_zero
    def warning(self, msg):
        self.logger.warning(msg)

    @staticmethod
    def exception(err):
        pt.exception(err)
        # if '{}'.format(err) == 'MPI_ERR_INTERN: internal error':
        #     # Known Issues: Allocating too much memory
        #     self.screen("Attempting to handle MPI error gracefully.\nAborting MPI...")
        #     pt.abort()

    def output(self, *args):
        pass


@contextmanager
def optional_write(file, mode, *args, openfn=None, **kwargs):
    # Note: this is suboptimal in that whatever write operations
    # are still performed ; this is negligible compared to the computation, at the moment.
    """If file is None, yields a dummy file object."""
    if file is None:
        return
    else:
        if openfn is None:
            openfn = gzip.open if file.endswith('.gz') else open
        with openfn(file, mode, *args, **kwargs) as open_file:
            with print_doing(f'Writing file "{file}"'):
                yield open_file


@contextmanager
def print_doing(msg, sep='', flush=True, end='', **kwargs):
    """Let the user know that the code is performing MSG. Also makes code more self-documenting."""
    start_time = datetime.now()
    print(msg, '...', sep=sep, flush=flush, end=end, **kwargs)
    yield
    print(f"Done! ({datetime.now()-start_time})")
