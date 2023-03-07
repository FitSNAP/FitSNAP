from fitsnap3lib.scrapers.scrape import Scraper
"""Methods you may or must override in new scrapers"""


class Template(Scraper):

    def __init__(self, name):
        super().__init__(name)

    # Scraper may override scrape_groups method
    def scrape_groups(self):
        """Need self.files and self.group_table"""
        pass

    # Scraper must override scrape_configs method
    def scrape_configs(self):
        """Generate and send (mutable) data to send to fitsnap"""
        pass
