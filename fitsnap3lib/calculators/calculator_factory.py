from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.calculators.calculator import Calculator
from fitsnap3lib.calculators.lammps_pace import LammpsPace
from fitsnap3lib.calculators.lammps_snap import LammpsSnap
from fitsnap3lib.calculators.basic_calculator import Basic


pt = ParallelTools()

def calculator(calculator_name):
    """Calculator Factory"""
    instance = search(calculator_name)
    pt.single_print("Using {} as FitSNAP calculator".format(calculator_name))
    instance.__init__(calculator_name)
    return instance


def search(calculator_name):
    instance = None
    print(Calculator.__subclasses__())
    for cls in Calculator.__subclasses__():
        if cls.__name__.lower() == calculator_name.lower():
            instance = Calculator.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in fitsnap calculators".format(calculator_name))
    else:
        return instance
