from fitsnap3.io.input import config


# TODO : Add file handling and screen handling
class Output:

    def __init__(self):
        self._screen = config.args.screen
        self._pscreen = config.args.pscreen
        self._logfile = config.args.logfile

    def output(self):
        pass


if __name__ == "fitsnap.io.input":
    output = Output()
