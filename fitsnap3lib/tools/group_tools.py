"""
This file houses functions that are useful for organization groups of training data.
"""
import random
import math

def make_table(group_settings):
    """
    Make a group table dictionary out of group settings dictionary.

    Returns a group table dictionary.
    """
    group_sections = group_settings.pop("group_sections")
    group_table = {}
    for k, v in group_settings.items():
        #print(f"{k} {v}")
        group_table[k] = {group_sections[i]: item for i, item in enumerate(v)}
    return group_table


def assign_validation(group_table, random_test = False):
    """
    Given a dictionary of group info, add another key for test bools.

    Args:
        group_table: Dictionary of group names. Must have keys "nconfigs" and "testing_size". 
        randomly: Boolean for whether to assign validation randomly (True) or to take last fraction 
                  of configurations in a group.

    Modifies the dictionary in place by adding another key "test_bools".
    """

    for name in group_table:
        nconfigs = group_table[name]["nconfigs"]
        assert("testing_size" in group_table[name])
        assert(group_table[name]["testing_size"] <= 1.0)

        if random_test:
            test_bools = [random.random() < group_table[name]["testing_size"] for i in range(0,nconfigs)]
            group_table[name]["test_bools"] = test_bools
        else:
            # Take last percentage of each group.
            ntest = math.ceil((group_table[name]["testing_size"]*nconfigs))
            ntrain = nconfigs - ntest
            test_bools = [i > ntrain for i in range(nconfigs)]
            group_table[name]["test_bools"] = test_bools