from fitsnap3.io.outputs.outputs import Output, pt
pt.get_subclasses(__name__, __file__, Output)


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
