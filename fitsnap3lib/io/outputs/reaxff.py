from fitsnap3lib.io.outputs.outputs import Output, optional_open
import os, threading
from datetime import datetime

class Reaxff(Output):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._io_lock = threading.Lock()

    # --------------------------------------------------------------------------------------------

    def output(self, coeffs, errors):
        def thread_safe_write():
            with self._io_lock:
                base, ext = os.path.splitext(self.config.sections["OUTFILE"].potential_name)
                timestamp = datetime.now().strftime("%Y%m%d-%H%M")
                with optional_open(f"{base}-{timestamp}{ext}", 'wt') as file:
                    file.write(coeffs)

        if self.pt._rank == 0:
            thread_safe_write()

    # --------------------------------------------------------------------------------------------
