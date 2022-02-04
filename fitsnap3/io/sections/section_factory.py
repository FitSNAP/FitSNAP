from fitsnap3.io.sections.sections import Section
from fitsnap3.io.error import ExitFunc
from fitsnap3.parallel_tools import ParallelTools

pt = ParallelTools()


def new_section(section, config, args):
    """Section Factory"""
    instance = search(section)
    try:
        instance.__init__(section, config, args)
    except ExitFunc:
        pass
    return instance


def search(section):
    instance = None
    for cls in Section.__subclasses__():
        if cls.__name__.lower() == section.lower():
            instance = Section.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in fitsnap sections".format(section))
    else:
        return instance
