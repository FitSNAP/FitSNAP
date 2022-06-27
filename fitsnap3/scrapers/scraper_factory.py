from fitsnap3.scrapers.scrape import Scraper, pt
pt.get_subclasses(__name__, __file__, Scraper)


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
