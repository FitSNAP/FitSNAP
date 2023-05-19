from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.outputs.outputs import Output
from fitsnap3lib.io.outputs.pace import Pace
from fitsnap3lib.io.outputs.snap import Snap
from fitsnap3lib.io.outputs.custom import Custom


#pt = ParallelTools()

def output(output_name, pt, cfg):
    """Output Factory"""
    instance = search(output_name)
    instance.__init__(output_name, pt, cfg)
    return instance


def output_factory(output_name):
    """Output Factory"""
    instance = search(output_name)
    instance.__init__(output_name)
    return instance


def search(output_name):
    instance = None
    for cls in Output.__subclasses__():
        if cls.__name__.lower() == output_name.lower():
            instance = Output.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in fitsnap outputs".format(output_name))
    else:
        return instance
