from pathlib import Path
from pkgutil import iter_modules
from importlib import import_module

alt_names = {"stress": "pressure",
             "virial": "pressure",
             "positions": "length",
             "position": "length",
             "forces": "force",
             "lattice": "length"}

name = __name__.split('.')
unit_dir = Path(__file__).resolve().parent
unit_type_dict = {}

for (_, unit_type, c) in iter_modules([unit_dir]):
    if unit_type != "conversion_finder":
        unit_type_dict[unit_type] = import_module(f"{'.'.join(name[:-1])}.{unit_type}")


def rename_unit(this_unit):
    this_unit = "_per_".join(this_unit.split("/"))
    return "_".join(this_unit.split("*"))


def rename_unit_type(a_unit_type):
    a_unit_type = a_unit_type.lower()
    if a_unit_type in alt_names:
        a_unit_type = alt_names[a_unit_type]
    return a_unit_type


def create_conversion(a_unit_type, unit_a, unit_b):
    a_unit_type = rename_unit_type(a_unit_type)
    unit_a = rename_unit(unit_a)
    unit_b = rename_unit(unit_b)

    try:
        the_unit_type = unit_type_dict[a_unit_type]
    except AttributeError:
        raise AttributeError("{} was not found in unit types".format(a_unit_type))

    try:
        numerator = getattr(the_unit_type, unit_a)
    except AttributeError:
        raise AttributeError("{} was not found in unit type {}".format(unit_a, a_unit_type))

    try:
        denominator = getattr(the_unit_type, unit_b)
    except AttributeError:
        raise AttributeError("{} was not found in unit type {}".format(unit_b, a_unit_type))

    return numerator/denominator

