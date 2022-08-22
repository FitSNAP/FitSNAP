from fitsnap3lib.io.input import Config
from fitsnap3lib.parallel_tools import ParallelTools
from contextlib import contextmanager
import gzip
from datetime import datetime
import logging


#config = Config()
#pt = ParallelTools()


class Output:

    def __init__(self, name):
        self.config = Config()
        self.pt = ParallelTools()
        self.name = name
        self._screen = self.config.args.screen
        self._pscreen = self.config.args.pscreen
        self._nscreen = self.config.args.nscreen
        self._logfile = self.config.args.log
        self._s2f = self.config.args.screen2file
        if self._s2f is not None:
            self.pt.set_output(self._s2f, ns=self._nscreen, ps=self._nscreen)
        self.logger = None
        if not logging.getLogger().hasHandlers():
            self.pt.pytest_is_true()
        if self._logfile is None:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.DEBUG, filename=self._logfile)
        self.logger = logging.getLogger(__name__)
        self.pt.set_logger(self.logger)

    def screen(self, *args, **kw):
        if self._pscreen:
            self.pt.all_print(*args, **kw)
        elif self._nscreen:
            self.pt.sub_print(*args, **kw)
        elif self._screen:
            self.pt.single_print(*args, **kw)
        else:
            pass

    #@pt.rank_zero
    def info(self, msg):
        @self.pt.rank_zero
        def decorated_info():
            self.logger.info(msg)
        decorated_info()

    #@pt.rank_zero
    def warning(self, msg):
        @self.pt.rank_zero
        def decorated_warning():
            self.logger.warning(msg)
        decorated_warning()

    @staticmethod
    def exception(self, err):
        self.pt.exception(err)
        # if '{}'.format(err) == 'MPI_ERR_INTERN: internal error':
        #     # Known Issues: Allocating too much memory
        #     self.screen("Attempting to handle MPI error gracefully.\nAborting MPI...")
        #     pt.abort()

    def output(self, *args):
        pass

    #@pt.rank_zero
    def write_errors(self, errors):
        @self.pt.rank_zero
        def decorated_write_errors():
            fname = self.config.sections["OUTFILE"].metric_file
            arguments = {}
            write_type = 'wt'
            if self.config.sections["OUTFILE"].metrics_style == "MD":
                # fname += '.md'
                function = errors.to_markdown
            elif self.config.sections["OUTFILE"].metrics_style == "CSV":
                # fname += '.csv'
                arguments['sep'] = ','
                arguments['float_format'] = "%.8f"
                function = errors.to_csv
            elif self.config.sections["OUTFILE"].metrics_style == "SSV":
                arguments['sep'] = ' '
                arguments['float_format'] = "%.8f"
                function = errors.to_csv
            elif self.config.sections["OUTFILE"].metrics_style == "JSON":
                # fname += '.json'
                function = errors.to_json
            elif self.config.sections["OUTFILE"].metrics_style == "DF":
                # fname += '.db'
                function = errors.to_pickle
                write_type = 'wb'
            else:
                raise NotImplementedError("Metric style {} not implemented".format(
                    self.config.sections["OUTFILE"].metrics_style))
            with optional_open(fname, write_type) as file:
                function(file, **arguments)
        decorated_write_errors()


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
