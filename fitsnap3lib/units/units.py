from fitsnap3lib.units.conversion_finder import create_conversion

conversions = {}


def convert(unit_type, unit_a=None, unit_b=None):
    if isinstance(unit_type, list):
        unit_type, unit_a, unit_b = [x for x in unit_type]

    try:
        return conversions[unit_type][unit_a][unit_b]
    except KeyError:
        if unit_type not in conversions:
            conversions[unit_type] = {}
            conversions[unit_type][unit_a] = {}
            conversions[unit_type][unit_a][unit_b] = create_conversion(unit_type, unit_a, unit_b)
        elif unit_a not in conversions[unit_type]:
            conversions[unit_type][unit_a] = {}
            conversions[unit_type][unit_a][unit_b] = create_conversion(unit_type, unit_a, unit_b)
        elif unit_b not in conversions[unit_type][unit_a]:
            conversions[unit_type][unit_a][unit_b] = create_conversion(unit_type, unit_a, unit_b)
        return conversions[unit_type][unit_a][unit_b]
