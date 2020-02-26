from fitsnap3.io.outputs.outputs import Output
"""Methods you may or must override in new output"""


class Template(Output):

    def __init__(self, name):
        super().__init__(name)

    # You must override this method
    def output(self):
        pass
