from fitsnap3.scrapers.scrape import Scraper
from fitsnap3.scrapers.template_scraper import Template
from fitsnap3.scrapers.json_scraper import Json


def scraper(scraper_name):
    """Section Factory"""
    instance = search(scraper_name)
    instance.__init__(scraper_name)
    return instance


def search(scraper_name):
    instance = None
    for cls in Scraper.__subclasses__():
        if cls.__name__.lower() == scraper_name.lower():
            instance = Scraper.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in fitsnap scrapers".format(scraper_name))
    else:
        return instance
