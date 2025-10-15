from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.calculators.calculator import Calculator
from fitsnap3lib.calculators.lammps_pace import LammpsPace
from fitsnap3lib.calculators.lammps_pyace import LammpsPyace
#from fitsnap3lib.calculators.pyace import PyACE
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

    def find_subclass_recursive(base_class, target_name):
        """Recursively search through all subclass levels"""
        # Check the current class
        if base_class.__name__.lower() == target_name.lower():
            return base_class
        
        # Check all direct subclasses
        for subclass in base_class.__subclasses__():
            result = find_subclass_recursive(subclass, target_name)
            if result is not None:
                return result
        
        return None
    
    # Find the target class recursively
    target_class = find_subclass_recursive(Calculator, calculator_name)
    
    if target_class is not None:
        instance = Calculator.__new__(target_class)
    
    if instance is None:
        raise IndexError("{} was not found in fitsnap calculators".format(calculator_name))
    else:
        return instance
