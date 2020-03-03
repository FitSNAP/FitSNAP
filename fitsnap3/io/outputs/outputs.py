from fitsnap3.io.input import config
from fitsnap3.parallel_tools import pt
from contextlib import contextmanager
import gzip
from datetime import datetime


# TODO : Add file handling for log file, use logger module
class Output:

    def __init__(self, name):
        self.name = name
        self._screen = config.args.screen
        self._pscreen = config.args.pscreen
        self._nscreen = config.args.nscreen
        self._logfile = config.args.log

    def screen(self, *args, **kw):
        if self._pscreen:
            print(*args)
        elif self._nscreen:
            pt.sub_print(*args, **kw)
        elif self._screen:
            pt.single_print(*args, **kw)
        else:
            pass

    @pt.rank_zero
    def log(self, *args):
        pass

    def error(self, err):
        pt.close_lammps()
        # Try to funnel all of the error message to a log or screen
        self.log("%s" % err)
        # self.screen("%s" % err)
        raise_err(err)

    def output(self, dummy):
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


@pt.rank_zero
def raise_err(err):
    raise err
