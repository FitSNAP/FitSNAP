from ..input import config
from ...parallel_tools import pt
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
        self._s2f = config.args.screen2file
        if self._s2f is not None:
            pt.set_output(self._s2f, ns=self._nscreen, ps=self._nscreen)
        self.logger = None
        if not logging.getLogger().hasHandlers():
            pt.pytest_is_true()
        if self._logfile is None:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.DEBUG, filename=self._logfile)
        self.logger = logging.getLogger(__name__)
        pt.set_logger(self.logger)

    def screen(self, *args, **kw):
        if self._pscreen:
            pt.all_print(*args, **kw)
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

    @pt.rank_zero
    def write_errors(self, errors):
        fname = config.sections["OUTFILE"].metric_file
        arguments = {}
        write_type = 'wt'
        if config.sections["OUTFILE"].metrics_style == "MD":
            # fname += '.md'
            function = errors.to_markdown
        elif config.sections["OUTFILE"].metrics_style == "CSV":
            # fname += '.csv'
            arguments['sep'] = ','
            arguments['float_format'] = "%.8f"
            function = errors.to_csv
        elif config.sections["OUTFILE"].metrics_style == "SSV":
            arguments['sep'] = ' '
            arguments['float_format'] = "%.8f"
            function = errors.to_csv
        elif config.sections["OUTFILE"].metrics_style == "JSON":
            # fname += '.json'
            function = errors.to_json
        elif config.sections["OUTFILE"].metrics_style == "DF":
            # fname += '.db'
            function = errors.to_pickle
            write_type = 'wb'
        else:
            raise NotImplementedError("Metric style {} not implemented".format(
                config.sections["OUTFILE"].metrics_style))
        with optional_open(fname, write_type) as file:
            function(file, **arguments)


@contextmanager
def optional_open(file, mode, *args, openfn=None, **kwargs):
    # Note: this is suboptimal in that whatever write operations
    # are still performed ; this is negligible compared to the computation, at the moment.
    """If file is None, yields a dummy file object."""
    if file is None:
        return
    else:
        if openfn is None:
            openfn = gzip.open if file.endswith('.gz') else open
        with openfn(file, mode, *args, **kwargs) as open_file:
            # with print_doing(f'Writing file "{file}"'):
            yield open_file


@contextmanager
def print_doing(msg, sep='', flush=True, end='', **kwargs):
    """Let the user know that the code is performing MSG. Also makes code more self-documenting."""
    start_time = datetime.now()
    print(msg, '...', sep=sep, flush=flush, end=end, **kwargs)
    yield
    print(f"Done! ({datetime.now()-start_time})")
