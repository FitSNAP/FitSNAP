"""Template Class"""
from fitsnap3.io.sections.sections import Section


class Default(Section):

    def __init__(self, name, config, args):
        # let parent hold config and args
        super().__init__(name, config, args)
        # run init methods to populate class
        # delete config and args to rely only on child's members
        self.delete()
