from fitsnap3.io.sections.sections import Section, output


class Scraper(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.scraper = self.get_value("SCRAPER", "scraper", "JSON")
        self.save_group_scrape = self.get_value("SCRAPER", "save_group_scrape", "None", "str")
        self.read_group_scrape = self.get_value("SCRAPER", "read_group_scrape", "None", "str")
        self.properties = {"Stress": ["pressure", "Metal", "Metal"],
                           "Lattice": ["length", "Metal", "Metal"],
                           "Energy": ["energy", "Metal", "Metal"],
                           "Positions": ["length", "Metal", "Metal"],
                           "Forces": ["force", "Metal", "Metal"]}
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
