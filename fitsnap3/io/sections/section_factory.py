from .sections import Section, pt
from ..error import ExitFunc
pt.get_subclasses(__name__, __file__, Section)


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
