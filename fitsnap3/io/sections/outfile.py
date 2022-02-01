from os import path
from .sections import Section
from ...parallel_tools import pt
from os import path


class Outfile(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        allowedkeys = ['output_style','metrics','potential','detailed_errors']
        for value_name in config['OUTFILE']:
            if value_name in allowedkeys: continue
            else:
                raise RuntimeError(">>> Found unmatched variable in OUTFILE section of input: ", value_name)
                #pt.single_print(">>> Found unmatched variable in OUTFILE section of input: ",value_name)

        self._outfile()
        self.output_style = self.get_value("OUTFILE", "output_style", "ORIGINAL")
        self.delete()

    def _outfile(self):
        self.metric_file = self.check_path(self.get_value("OUTFILE", "metrics", "fitsnap_metrics.md"))
        self.potential_name = self.check_path(self.get_value("OUTFILE", "potential", "fitsnap_potential"))
        return
