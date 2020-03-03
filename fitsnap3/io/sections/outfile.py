from os import path
from fitsnap3.io.sections.sections import Section


class Outfile(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self._check_relative()
        self._outfile()
        self.output_style = self._config.get("OUTFILE", "output_style", fallback="ORIGINAL")
        self.delete()

    def _outfile(self):
        self.config_file = self._config.get("OUTFILE", "configs", fallback="fitsnap_configs.pkl.gz")
        self._check_path(self.config_file)
        self.metric_file = self._config.get("OUTFILE", "metrics", fallback="fitsnap_metrics.csv")
        self._check_path(self.metric_file)
        self.potential_name = self._config.get("OUTFILE", "potential", fallback="fitsnap_potential")
        self._check_path(self.potential_name)
        return

    def _check_relative(self):
        if self._args.relative:
            self.base_path, _ = path.split(self._args.infile)
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
            if self._args.overwrite and path.exists(element):
                raise FileExistsError(f"File {element} already exists.")