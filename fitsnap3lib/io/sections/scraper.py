from fitsnap3lib.io.sections.sections import Section
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.parallel_output import Output


pt = ParallelTools()
output = Output()


class Scraper(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['scraper', 'save_group_scrape', 'read_group_scrape', 'property_array',
        'vasp_use_TOTEN','vasp_json_pathname','vasp_ignore_incomplete','vasp_ignore_jsons',
        'vasp_unconverged_label','vasp_xml_only','vasp_outcar_only']
        self._check_section()

        self.scraper = self.get_value("SCRAPER", "scraper", "JSON")
        self.save_group_scrape = self.get_value("SCRAPER", "save_group_scrape", "None", "str")
        self.read_group_scrape = self.get_value("SCRAPER", "read_group_scrape", "None", "str")
        self.properties = {"Stress": ["pressure", "Metal", "Metal"],
                           "Lattice": ["length", "Metal", "Metal"],
                           "Energy": ["energy", "Metal", "Metal"],
                           "Positions": ["length", "Metal", "Metal"],
                           "Forces": ["force", "Metal", "Metal"]}

        ## Variables to configure VASP scraper (all optional)
        self.vasp_use_TOTEN = self.get_value("SCRAPER", "vasp_use_TOTEN", "False", "bool")
        self.vasp_json_pathname = self.get_value("SCRAPER", "vasp_json_pathname", "vJSON", "str")
        self.vasp_ignore_incomplete = self.get_value("SCRAPER", "vasp_ignore_incomplete", "1", "bool")
        self.vasp_ignore_jsons = self.get_value("SCRAPER", "vasp_ignore_jsons", "0", "bool")
        self.vasp_xml_only = self.get_value("SCRAPER", "vasp_xml_only", "0", "bool")  ## for testing
        self.vasp_outcar_only = self.get_value("SCRAPER", "vasp_outcar_only", "0", "bool")  ## for testing
        self.vasp_unconverged_label = self.get_value("SCRAPER", "vasp_unconverged_label", "UNCONVERGED", "str")
        temp_array = self.get_value("SCRAPER", "property_array", "None", "str")
        if temp_array != "None":
            temp_array = temp_array.replace("=", "").replace(":", "").replace(";", "\n").split("\n")
            for item in temp_array:
                if item == '':
                    continue
                elements = item.split()
                self.properties[elements[0].capitalize()] = elements[1:]
        # TODO: implement unit systems
        # self.unit_system = self.get_value("SCRAPER", "unit_system", "None", "str")
        self.delete()
