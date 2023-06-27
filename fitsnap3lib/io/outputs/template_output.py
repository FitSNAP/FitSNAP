from fitsnap3lib.io.outputs.outputs import Output
"""Methods you may or must override in new output"""


class Template(Output):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)

    # You must override this method
    def output(self):
        pass
