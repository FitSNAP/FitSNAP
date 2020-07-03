from os import path
from fitsnap3.io.sections.sections import Section


class Outfile(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self._check_relative()
        self._outfile()
        self.output_style = self.get_value("OUTFILE", "output_style", "ORIGINAL")
        self.delete()

    def _outfile(self):
        self.config_file = self._check_path(self.get_value("OUTFILE", "configs", "fitsnap_configs.pkl.gz"))
        self.metric_file = self._check_path(self.get_value("OUTFILE", "metrics", "fitsnap_metrics.csv"))
        self.potential_name = self._check_path(self.get_value("OUTFILE", "potential", "fitsnap_potential"))
        return

    def _check_relative(self):
        if self._args.relative:
            self.base_path = self._get_relative_directory(self)
        else:
            self.base_path = None

    def _check_path(self, name):
        if self.base_path is not None:
            name = path.join(self.base_path, name)
        else:
            name = name
        if self._args.overwrite is None:
            return name
        names = [name, name + '.snapparam', name + '.snapcoeff']
        for element in names:
            if not self._args.overwrite and path.exists(element):
                raise FileExistsError(f"File {element} already exists.")
        return name
