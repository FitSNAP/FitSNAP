
import os, re
from itertools import islice

with open('../src/examples.rst', 'w') as rst:

    print( '\nExamples\n=========\n\n', file=rst )
    
    with os.scandir('../../examples') as examples:

        sorted_examples = sorted([e for e in examples if e.is_dir()], key=lambda e: e.name)
        
        for i, e in enumerate(sorted_examples):
            #print(e)

            if i>0: print("--------\n", file=rst )
            print(f"{e.name}\n{''.join(['-']*len(e.name))}\n", file=rst )

            if os.path.isfile(readme_path := f"../../examples/{e.name}/README.md"):
                print(f".. include:: {readme_path}\n", file=rst )
                #print(f"  :parser: myst_parser.sphinx_\n", file=rst )
            else:
                print(f"See ``examples/{e.name}/`` for details.\n\n", file=rst )
            
