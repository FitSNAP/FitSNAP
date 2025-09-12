from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.calculators.calculator import Calculator
from fitsnap3lib.calculators.lammps_pace import LammpsPace
from fitsnap3lib.calculators.pyace import PyACE
from fitsnap3lib.calculators.lammps_snap import LammpsSnap
#from fitsnap3lib.calculators.basic_calculator import Basic
from fitsnap3lib.calculators.lammps_custom import LammpsCustom


#pt = ParallelTools()

def calculator(calculator_name, pt, cfg):
    """Calculator Factory"""
    instance = search(calculator_name)
    #pt = ParallelTools()
    if cfg.args.verbose:
        pt.single_print("Using {} as FitSNAP calculator".format(calculator_name))
    
    instance.__init__(calculator_name, pt, cfg)
    return instance


def search(calculator_name):
    instance = None

    # First check direct subclasses of Calculator
    for cls in Calculator.__subclasses__():
        if cls.__name__.lower() == calculator_name.lower():
            instance = Calculator.__new__(cls)
            break
    
    # If not found, check subclasses of subclasses (for LammpsBase hierarchy)
    if instance is None:
        for cls in Calculator.__subclasses__():
            for cls2 in cls.__subclasses__():
                if cls2.__name__.lower() == calculator_name.lower():
                    instance = Calculator.__new__(cls2)
                    break
            if instance is not None:
                break

    if instance is None:
        raise IndexError("{} was not found in fitsnap calculators".format(calculator_name))
    else:
        return instance
