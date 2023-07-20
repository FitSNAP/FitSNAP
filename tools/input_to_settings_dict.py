"""
Script to convert a FitSNAP input file to a dictionary
Dictonary is printed to stdout and can be copied directly into a FitSNAP script
If write_json = True, also writes the dictionary to a JSON file
"""
import os, configparser, json

infile = "fitsnap.in"
write_json = True

if not os.path.exists(infile):
  print(f"Could not locate FitSNAP input file '{infile}'!\nChange variable 'infile' in script and run again.")
  exit()

## you can use the next 3 lines to generate the settings dictionary directly inside a python script
c = configparser.ConfigParser()
c.read(infile)
settings = {s:dict(c.items(s)) for s in c.sections()}
    
## you can use this next line to print out a pretty looking version of the dictionary as a JSON
## it should be valid code that you can copy-paste into a python script and run
print(json.dumps(settings, indent=4))

## if you want, can also write to a JSON file 
## see note at bottom about reading the JSON file in FitSNAP or another program
if write_json:
  outfile = f"{infile}.json"
  with open(outfile, 'w') as f:
    json.dump(settings, f, indent=4)
  print(f"Wrote '{infile}' dict to output file: ", outfile)

## to read JSON file, use:
## with open(outfile, 'r') as f:
##  new_dict_object = json.loads(f.read())
