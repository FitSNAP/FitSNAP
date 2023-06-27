"""Template Class"""
from fitsnap3lib.io.sections.sections import Section


class Default(Section):

    def __init__(self, name, config, pt,infile, args):
        # let parent hold config and args
        super().__init__(name, config, pt, infile, args)
        # run init methods to populate class
        # delete config and args to rely only on child's members
        self.delete()
