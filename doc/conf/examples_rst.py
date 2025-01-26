
import os, re
from itertools import islice

with open('../src/examples.rst', 'w') as rst:

    print( '\nExamples\n=========\n\n', file=rst )

    #.. include:: md/Compiling-and-running-on-a-local-computer.md
    #  :parser: myst_parser.sphinx_
    
    with os.scandir('../../examples') as examples:
        for e in sorted([e for e in examples if e.is_dir()], key=lambda e: e.name):
            #print(e)
            print(f"--------\n\n{e.name}\n{''.join(['-']*len(e.name))}\n", file=rst )

            if os.path.isfile(readme_path := f"../../examples/{e.name}/README.md"):
                print(f".. include:: {readme_path}", file=rst )
                print(f"  :parser: myst_parser.sphinx_\n", file=rst )
            else:
                print(f"See ``examples/{e.name}/`` for details.\n\n", file=rst )
            
