"""
---> Script to convert a FitSNAP input file to a dictionary <---
(complements settings_dict_to_input.py)
Dictonary is printed to stdout which can be copied into FitSNAP library scripts, and can optionally be written to a Python file.

There are two file formats that you can write to, *.py and *.json.
The output file name will be automatically generated based on the infile name.
Optionally, you can also choose your own "output_label" (no need to add file formats, those are added later).
If write_python_dict = True, writes the dictionary as a variable 'input_dict' to a Python file, which can be copied or imported into any FitSNAP script.
If write_json = True, writes the dictionary to a JSON file, which can be copied or read into any FitSNAP script using the native Python json module.

"""
import os, configparser, json

# ------------------- user settings

# REQUIRED: name of FitSNAP input 
infile = "SNAP_Ta.in"

# OPTIONAL: write to a Python file as a variable 'input_dict' or a standard JSON file
write_python_dict = True
write_json = True

# OPTIONAL: choose a label for *.json or *.py output (file format will be appended)
# if this is left empty ('' or None or 0 or [] or ...), a name will be automatically generated
outfile_label = "" 

# ------------------- run script

# if user doesn't choose a name for the output file, use infile label
if not outfile_label:
  infile_label = infile.replace(".","").replace("in","")
  outfile_label = f'settings_{infile_label}'

if not os.path.exists(infile):
  print(f"Could not locate FitSNAP input file '{infile}'!\nChange variable 'infile' in script and run again.")
  exit()

# you can use the next 3 lines to generate the settings dictionary directly inside a python script
# the c.optionxform = str line makes sure that string cases are conserved on read-in
c = configparser.ConfigParser()
c.optionxform = str 
c.read(infile)
settings = {s:dict(c.items(s)) for s in c.sections()}

# you can use this next line to print out a pretty looking version of the settings dictionary
# it should be valid code that you can copy-paste into a python script and run
print(json.dumps(settings, indent=4))

# write Python file with 'input_dict' dictionary variable
# this can be copied from the file, or imported as a module, e.g.:
# from settings_fitsnap-in import input_dict
if write_python_dict:
  outfile = f"{outfile_label}.py"
  with open(outfile, 'w') as f:
    f.write("# Settings dictionary for FitSNAP input, to use:\n")
    f.write(f"# from {outfile_label} import input_dict\n")
    f.write("input_dict = \\")
    f.write("\n")
    json.dump(settings, f, indent=4)
  print(f"Wrote '{infile}' dict to output file: ", outfile)

## if you want, can also write to a JSON file 
## see note at bottom about reading the JSON file in FitSNAP or another program
if write_json:
  outfile = f"{infile}.json"
  with open(outfile, 'w') as f:
    json.dump(settings, f, indent=4)
  print(f"Wrote '{infile}' dict to output file: ", outfile)

# to read JSON file in any program, use:
# with open(outfile, 'r') as f:
#  new_dict_object = json.loads(f.read(), strict=False)
#
# Note: sometimes need the strict=False argument cause json.loads borks on newlines
