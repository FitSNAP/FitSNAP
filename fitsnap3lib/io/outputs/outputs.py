#from fitsnap3lib.io.input import Config
#from fitsnap3lib.parallel_tools import ParallelTools
from contextlib import contextmanager
import gzip
from datetime import datetime
import logging


#config = Config()
#pt = ParallelTools()


class Output:

    def __init__(self, name, pt, config):
        self.config = config #Config()
        self.pt = pt #ParallelTools()
        self.name = name
        self._screen = self.config.args.screen
        self._pscreen = self.config.args.pscreen
        self._nscreen = self.config.args.nscreen
        self._logfile = self.config.args.log
        self._s2f = self.config.args.screen2file
        self._tarball = self.config.args.tarball
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
    
    def write_lammps(self, *args):
        """Parent class function for writing LAMMPS-ready potential files."""
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

        # Don't write errors if using scalapack.
        if not self.config.sections["SOLVER"].true_multinode: 
            # Don't write errors if `errors` is a list (default is empty list).
            if type(errors) != type([]):
                decorated_write_errors()

    def write_errors_nn(self, errors):
        """ 
        Write errors for nonlinear fits. 
        
        Args:
            errors : sequence of dictionaries (mae_f, mae_e, rmse_f, rmse_e, count_train, count_test)
        """
        @self.pt.rank_zero
        def decorated_write_errors_nn():
            mae_f = errors[0]
            mae_e = errors[1]
            rmse_f = errors[2]
            rmse_e = errors[3]
            count_train = errors[4]
            count_test = errors[5]
            fname = self.config.sections["OUTFILE"].metric_file
            fh = open(fname, 'w')

            # Find longest group name for formatting.
            longest = max([len(g) for g in self.config.sections['GROUPS'].group_table])

            colnames = f"{'Group' : <{longest}}  {'Train/Test' : ^10}  {'Property' : ^10}  {'Count' : ^10} {'MAE' : ^10}  {'RMSE' : ^10}\n"
            fh.write(colnames)
            out  = f"{'*ALL' : <{longest}} {'Train' : ^12} {'Energy' : ^10} {count_train['*ALL']['nconfigs'] : ^10} {mae_e['*ALL']['train'] : ^10.3e} {rmse_e['*ALL']['train'] : >10.3e}\n"
            out += f"{'*ALL' : <{longest}} {'Train' : ^12} {'Force' : ^10} {count_train['*ALL']['natoms'] : ^10} {mae_f['*ALL']['train'] : ^10.3e} {rmse_f['*ALL']['train'] : >10.3e}\n"
            out += f"{'*ALL' : <{longest}} {'Test' : ^12} {'Energy' : ^10} {count_test['*ALL']['nconfigs'] : ^10} {mae_e['*ALL']['test'] : ^10.3e} {rmse_e['*ALL']['test'] : >10.3e}\n"
            out += f"{'*ALL' : <{longest}} {'Test' : ^12} {'Force' : ^10} {count_test['*ALL']['natoms'] : ^10} {mae_f['*ALL']['test'] : ^10.3e} {rmse_f['*ALL']['test'] : >10.3e}\n"
            for group in self.config.sections['GROUPS'].group_table:
                out += f"{group : <{longest}} {'Train' : ^12} {'Energy' : ^10} {count_train[group]['nconfigs'] : ^10} {mae_e[group]['train'] : ^10.3e} {rmse_e[group]['train'] : >10.3e}\n"
                out += f"{group : <{longest}} {'Train' : ^12} {'Force' : ^10} {count_train[group]['natoms'] : ^10} {mae_f[group]['train'] : ^10.3e} {rmse_f[group]['train'] : >10.3e}\n"
                out += f"{group : <{longest}} {'Test' : ^12} {'Energy' : ^10} {count_test[group]['nconfigs'] : ^10} {mae_e[group]['test'] : ^10.3e} {rmse_e[group]['test'] : >10.3e}\n"
                out += f"{group : <{longest}} {'Test' : ^12} {'Force' : ^10} {count_test[group]['natoms'] : ^10} {mae_f[group]['test'] : ^10.3e} {rmse_f[group]['test'] : >10.3e}\n"
            fh.write(out)
            
            fh.close()

            """
            colnames = f"{'Group' : <{longest}}  {'Train/Test' : ^10}  {'Force MAE' : ^10}  {'Energy MAE' : ^10}  {'Force RMSE' : ^10}  {'Energy RMSE' : >10}\n"
            fh.write(colnames)
            line = f"{'*ALL' : <{longest}}  {'Train' : ^10}  {mae_f['*ALL']['train'] : ^10.3e}  {mae_e['*ALL']['train'] : ^10.3e}  {rmse_f['*ALL']['train'] : ^10.3e}  {rmse_e['*ALL']['train'] : >10.3e}\n"
            line += f"{ '' : <{longest}}  {'Test' : ^10}  {mae_f['*ALL']['test'] : ^10.3e}  {mae_e['*ALL']['test'] : ^10.3e}  {rmse_f['*ALL']['test'] : ^10.3e}  {rmse_e['*ALL']['test'] : >10.3e}\n"
            fh.write(line)
            for group in self.config.sections['GROUPS'].group_table:
                line = f"{group : <{longest}}  {'Train' : ^10}  {mae_f[group]['train'] : ^10.3e}  {mae_e[group]['train'] : ^10.3e}  {rmse_f[group]['train'] : ^10.3e}  {rmse_e[group]['train'] : >10.3e}\n"
                line += f"{ '' : <{longest}}  {'Test' : ^10}  {mae_f[group]['test'] : ^10.3e}  {mae_e[group]['test'] : ^10.3e}  {rmse_f[group]['test'] : ^10.3e}  {rmse_e[group]['test'] : >10.3e}\n"
                fh.write(line)
            fh.close()
            """
        decorated_write_errors_nn()


@contextmanager
def optional_open(file, mode, *args, openfn=None, **kwargs):
    # NOTE: this is suboptimal in that whatever write operations
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
