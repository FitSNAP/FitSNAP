from fitsnap3.io.sections.sections import Section


class Scraper(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        if "SCRAPER" not in self._config:
            self.scraper = "JSON"
        else:
            self.scraper = self._config.get("SCRAPER", "scraper", fallback="JSON")
        self.delete()
