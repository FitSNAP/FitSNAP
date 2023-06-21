
## you can use the first 4 lines to just generate the settings dictionary directly inside a python script
import configparser
c = configparser.ConfigParser()
c.read("fitsnap.in")
settings = {s:dict(c.items(s)) for s in c.sections()}

## you can use these last 2 lines to print out a pretty looking version of the dictionary
## it should be valid code that you can copy-paste into a python script and run
import json
print(json.dumps(d, indent=4))
