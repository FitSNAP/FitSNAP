from fitsnap3lib.io.outputs.outputs import Output, optional_open
import os, threading
import pandas as pd
from datetime import datetime

class Reaxff(Output):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._io_lock = threading.Lock()
        dataPath = self.config.sections["PATH"].dataPath
        potential_name = self.config.sections["OUTFILE"].potential_name
        self.base = f"{dataPath}/{os.path.splitext(os.path.basename(potential_name))[0]}"
        self.ext  = os.path.splitext(potential_name)[1]

    # --------------------------------------------------------------------------------------------

    def output(self, coeffs, errors):
        def thread_safe_write():
            with self._io_lock:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M")
                with optional_open(f"{self.base}-{timestamp}{self.ext}", 'wt') as file:
                    file.write(coeffs)
                if isinstance(errors, pd.DataFrame) and not errors.empty:
                    errors.to_csv(f"{self.base}-{timestamp}.csv", index=False)
                    #errors.to_feather("{base}-{timestamp}.feather", index=False)


        if self.pt._rank == 0:
            thread_safe_write()

    # --------------------------------------------------------------------------------------------
