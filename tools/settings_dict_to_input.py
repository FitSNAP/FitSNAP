"""
---> Script to convert a FitSNAP settings dictionary to an input file <---
(complements input_to_settings_dict.py)
This script can take a settings dictionary (copied directly here, or read from a JSON or Python* file) and writes a standard FitSNAP input file.
  
Optionally, you can also choose your own "output_label" (no need to add file formats, those are added later).
If "output_label" is left empty, the output file name will be simply 'fitsnap.in'
If print_to_screen = True, prints the new input file to screen.

*Note: Python files must contain only the FitSNAP settings dictionary, written as a variable - i.e. input_dict = {...}!
"""
import os, configparser, json

# ------------------- user settings
# REQUIRED: FitSNAP settings dictionary for input
input_dict = "settings_SNAP_Ta.py"
# NOTE: two formats are accepted - either:
#   - a FitSNAP settings dictionary, OR
#   - a string such as "fitsnap_settings.json" OR "fitsnap_settings.py"
# Note that the JSON or Python file must contain ONLY the input_dict object!

# OPTIONAL: choose a label for your file, i.e. my_fancy_file (no file format needed)
# if this is left empty ('' or None or 0 or [] or ...), a name will be automatically generated
outfile_label = "" 

# OPTIONAL: print new file to screen
print_to_screen = True

# ------------------- run script

# if user doesn't choose a name for the output file, use infile label
if not outfile_label:
  outfile_label = 'fitsnap'

outfile = f'{outfile_label}.in'

# If no input_dict is specified, check if an input file was specified.
if type(input_dict) == str:
  if not os.path.exists(input_dict):
    print(f"Could not locate any infile '{input_dict}'!\nChange variable 'infile' in script and run again.")
    exit()
  else:
    with open(input_dict, 'r') as f:
      if '.json' in input_dict:
        txt = f.read()
      elif ".py" in input_dict:
        txt0 = f.read()
        start_dict_idx = txt0.find("{")
        txt = txt0[start_dict_idx:]
        
    input_dict = json.loads(txt, strict=False)

# the configparser formats the dictionary for proper FitSNAP read-in
# the c.optionxform = str line makes sure that string cases are conserved on read-in
c = configparser.ConfigParser()
c.optionxform = str 
for k, v in input_dict.items():
  c[k] = v

# have the configparser write the file to disk
with open(outfile, 'w') as f:
  c.write(f)
print(f"Wrote input_dict to output file: ", outfile)

# this prints the file line-by-line to screen (stdout)
if print_to_screen:
  print(f"Printing FitSNAP input file: {outfile}")
  with open(outfile, 'r') as f:
    txt = f.readlines()
  for line in txt:
    print(line.strip())
