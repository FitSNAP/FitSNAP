from fitsnap3lib.io.outputs.outputs import Output, optional_open
from datetime import datetime
import numpy as np
import itertools
import pickle

#config = Config()
#pt = ParallelTools()

try:

    from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import AcePot
    from fitsnap3lib.lib.sym_ACE.wigner_couple import *
    from fitsnap3lib.lib.sym_ACE.clebsch_couple import *

    class Pace(Output):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)
            #self.config = Config()
            #self.pt = ParallelTools()

        def output(self, coeffs, errors):
            if (self.config.sections["CALCULATOR"].nonlinear):
                self.write_nn(errors)
            else:
                new_coeffs = None
                # new_coeffs = pt.combine_coeffs(coeffs)
                if new_coeffs is not None:
                    coeffs = new_coeffs
                self.write(coeffs, errors)

        def write_lammps(self, coeffs):
            """
            Write LAMMPS ready ACE files.

            Args:
                coeffs: list of linear model coefficients.
            """
            if self.config.sections["EXTRAS"].only_test != 1:
                if self.config.sections["CALCULATOR"].calculator not in ["LAMMPSPACE", "PYACE"]:
                    raise TypeError("PACE output style must be paired with LAMMPSPACE or PYACE calculator")
                
                # For ACE section, write both .acecoeff and potential files
                if "ACE" in self.config.sections and hasattr(self.config.sections["ACE"], 'blank2J'):
                    with optional_open(self.config.sections["OUTFILE"].potential_name and
                                      self.config.sections["OUTFILE"].potential_name + '.acecoeff', 'wt') as file:
                        file.write(_to_coeff_string(coeffs, self.config))
                    self.write_potential(coeffs)
                    with optional_open(self.config.sections["OUTFILE"].potential_name and self.config.sections["OUTFILE"].potential_name + '.mod', 'wt') as file:
                        file.write(_to_potential_file(self.config))
                # For PYACE section, only write potential files (skip .acecoeff)
                elif "PYACE" in self.config.sections:
                    self.pt.single_print("PYACE detected: Skipping .acecoeff output, generating .yace potential file only")
                    self.write_potential(coeffs)
                    with optional_open(self.config.sections["OUTFILE"].potential_name and self.config.sections["OUTFILE"].potential_name + '.mod', 'wt') as file:
                        file.write(_to_potential_file(self.config))
                else:
                    raise RuntimeError("No supported ACE or PYACE section found for output")

        #@pt.rank_zero
        def write(self, coeffs, errors):
            @self.pt.rank_zero
            def decorated_write():
                """
                if self.config.sections["EXTRAS"].only_test != 1:
                    if self.config.sections["CALCULATOR"].calculator != "LAMMPSPACE":
                        raise TypeError("PACE output style must be paired with LAMMPSPACE calculator")
                    with optional_open(self.config.sections["OUTFILE"].potential_name and
                                      self.config.sections["OUTFILE"].potential_name + '.acecoeff', 'wt') as file:
                        file.write(_to_coeff_string(coeffs, self.config))
                    self.write_potential(coeffs)
                """
                self.write_lammps(coeffs)
                self.write_errors(errors)
                # Generate validation notebook if validation option is specified
                if hasattr(self.config.sections["OUTFILE"], 'validation') and self.config.sections["OUTFILE"].validation:
                    self.write_validation_notebook(errors)
            decorated_write()

        def write_nn(self, errors):
            """ 
            Write output for nonlinear fits. 
            
            Args:
                errors : sequence of dictionaries (group_mae_f, group_mae_e, group_rmse_e, group_rmse_f)
            """
            @self.pt.rank_zero
            def decorated_write():
                # TODO: Add mliap decriptor writing when LAMMPS implementation of NN-ACE is complete.
                self.write_errors_nn(errors)
                # Generate validation notebook if validation option is specified
                if hasattr(self.config.sections["OUTFILE"], 'validation') and self.config.sections["OUTFILE"].validation:
                    self.write_validation_notebook_nn(errors)
            decorated_write()

        #@pt.sub_rank_zero
        def read_fit(self):
            @self.pt.sub_rank_zero
            def decorated_read_fit():
                assert NotImplementedError("read fit for pace potentials not implemented")
            decorated_read_fit()

        def write_validation_notebook(self, errors):
            """
            Generate a validation Jupyter notebook with the errors DataFrame and table generation code.
            
            Args:
                errors: pandas DataFrame containing metrics data
            """
            import json
            import io
            from contextlib import redirect_stdout
            
            if type(errors) == type([]):
                return
                
            # Extract base name for the notebook
            base_name = self.config.sections["OUTFILE"].potential_name or "fitsnap_potential"
            notebook_name = f"{base_name}_validation.ipynb"
            
            # Create notebook structure
            notebook = {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": "python",
                        "version": "3.8.0"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            # Add title cell
            title_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# FitSNAP Validation Report\n",
                    "\n",
                    f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                    f"Configuration hash: {self.config.hash}\n",
                    "\n"
                ]
            }
            notebook["cells"].append(title_cell)
            
            # Generate DataFrame code and table generation code
            code_lines = self._generate_table_code(errors)
            
            # Execute the code to get the HTML output
            html_output = self._execute_code(code_lines)
            
            # Create code cell with executed output
            code_cell = {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": code_lines,
                "outputs": [
                    {
                        "data": {
                            "text/html": [html_output],
                            "text/plain": ["<IPython.core.display.HTML object>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    }
                ]
            }
            
            notebook["cells"].append(code_cell)
            
            # Write notebook file
            with open(notebook_name, 'w') as f:
                json.dump(notebook, f, indent=2)
            
            self.pt.single_print(f"Generated validation notebook: {notebook_name}")

        def _generate_table_code(self, errors):
            """
            Generate Python code that recreates the DataFrame and creates HTML tables.
            """
            import pandas as pd
            
            # Convert DataFrame to code
            df_repr = self._dataframe_to_code(errors)
            
            code_lines = [
                "import pandas as pd\n",
                "from IPython.display import HTML, display\n",
                "\n",
                "# FitSNAP validation metrics DataFrame\n",
                f"errors = {df_repr}\n",
                "\n",
                "# Generate HTML tables\n",
                "def create_booktabs_table(data, title):\n",
                "    if len(data) == 0:\n",
                "        return f'<p>No {title.lower()} data available.</p>'\n",
                "    \n",
                "    # Process data\n",
                "    rows = []\n",
                "    for idx, row in data.iterrows():\n",
                "        if hasattr(idx, '__len__') and len(idx) >= 3:\n",
                "            group = idx[0]\n",
                "            weight_type = idx[1]\n",
                "            train_test = idx[2]\n",
                "            rows.append({\n",
                "                'group': group,\n",
                "                'weight': weight_type,\n",
                "                'split': train_test,\n",
                "                'count': int(row.get('ncount', 0)),\n",
                "                'mae': float(row.get('mae', 0)),\n",
                "                'rmse': float(row.get('rmse', 0)),\n",
                "                'rsq': float(row.get('rsq', 0))\n",
                "            })\n",
                "    \n",
                "    # Sort: groups first, then ALL rows\n",
                "    group_rows = [r for r in rows if r['group'] != '*ALL']\n",
                "    all_rows = [r for r in rows if r['group'] == '*ALL']\n",
                "    group_rows.sort(key=lambda x: (x['group'], x['weight'], x['split']))\n",
                "    all_rows.sort(key=lambda x: (x['weight'], x['split']))\n",
                "    \n",
                "    # Mark separator for ALL rows\n",
                "    if group_rows and all_rows:\n",
                "        all_rows[0]['separator'] = True\n",
                "    \n",
                "    table_data = group_rows + all_rows\n",
                "    \n",
                "    # Generate HTML\n",
                "    html = f'''<div style=\"margin: 20px 0; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;\">\n'''\n",
                "    html += f'''    <h3 style=\"color: #333; border-bottom: 2px solid #333; padding-bottom: 10px;\">{title} Metrics</h3>\n'''\n",
                "    html += '''    <table style=\"border-collapse: collapse; width: 100%; border-top: 2px solid #000; border-bottom: 2px solid #000; margin: 10px 0; background-color: #fff;\">\n'''\n",
                "    html += '''        <thead style=\"border-bottom: 1px solid #000; background-color: #f8f9fa;\">\n'''\n",
                "    html += '''            <tr>\n'''\n",
                "    html += '''                <th style=\"padding: 12px 15px; text-align: left; font-weight: 600; color: #333;\">Group</th>\n'''\n",
                "    html += '''                <th style=\"padding: 12px 15px; text-align: center; font-weight: 600; color: #333;\">Train/Test</th>\n'''\n",
                "    html += '''                <th style=\"padding: 12px 15px; text-align: right; font-weight: 600; color: #333;\">Count</th>\n'''\n",
                "    html += '''                <th style=\"padding: 12px 15px; text-align: right; font-weight: 600; color: #333;\">MAE</th>\n'''\n",
                "    html += '''                <th style=\"padding: 12px 15px; text-align: right; font-weight: 600; color: #333;\">RMSE</th>\n'''\n",
                "    html += '''                <th style=\"padding: 12px 15px; text-align: right; font-weight: 600; color: #333;\">R²</th>\n'''\n",
                "    html += '''            </tr>\n'''\n",
                "    html += '''        </thead>\n'''\n",
                "    html += '''        <tbody>\n'''\n",
                "    \n",
                "    for i, row in enumerate(table_data):\n",
                "        if row['group'] == '*ALL':\n",
                "            style = \"font-weight: bold; background-color: #e9ecef;\"\n",
                "            group_display = \"ALL\"\n",
                "        else:\n",
                "            style = \"background-color: #fff;\" if i % 2 == 0 else \"background-color: #f8f9fa;\"\n",
                "            group_display = row['group']\n",
                "        \n",
                "        sep_style = \"border-top: 1px solid #000;\" if row.get('separator') else \"\"\n",
                "        \n",
                "        html += f'''            <tr style=\"{style}\">\n'''\n",
                "        html += f'''                <td style=\"padding: 10px 15px; {sep_style}\">{group_display}</td>\n'''\n",
                "        html += f'''                <td style=\"padding: 10px 15px; text-align: center; {sep_style}\">{row['split']}</td>\n'''\n",
                "        html += f'''                <td style=\"padding: 10px 15px; text-align: right; {sep_style}\">{row['count']:,}</td>\n'''\n",
                "        html += f'''                <td style=\"padding: 10px 15px; text-align: right; font-family: monospace; {sep_style}\">{row['mae']:.6f}</td>\n'''\n",
                "        html += f'''                <td style=\"padding: 10px 15px; text-align: right; font-family: monospace; {sep_style}\">{row['rmse']:.6f}</td>\n'''\n",
                "        html += f'''                <td style=\"padding: 10px 15px; text-align: right; font-family: monospace; {sep_style}\">{row['rsq']:.6f}</td>\n'''\n",
                "        html += '''            </tr>\n'''\n",
                "    \n",
                "    html += '''        </tbody>\n'''\n",
                "    html += '''    </table>\n'''\n",
                "    html += '''</div>'''\n",
                "    \n",
                "    return html\n",
                "\n",
                "# Separate by property type and create tables\n",
                "energy_data = errors[errors.index.get_level_values(-1) == 'Energy'] if hasattr(errors.index, 'get_level_values') else errors[errors['Property'] == 'Energy'] if 'Property' in errors.columns else errors[errors.index.str.contains('Energy', na=False)] if hasattr(errors.index, 'str') else pd.DataFrame()\n",
                "force_data = errors[errors.index.get_level_values(-1) == 'Force'] if hasattr(errors.index, 'get_level_values') else errors[errors['Property'] == 'Force'] if 'Property' in errors.columns else errors[errors.index.str.contains('Force', na=False)] if hasattr(errors.index, 'str') else pd.DataFrame()\n",
                "stress_data = errors[errors.index.get_level_values(-1) == 'Stress'] if hasattr(errors.index, 'get_level_values') else errors[errors['Property'] == 'Stress'] if 'Property' in errors.columns else errors[errors.index.str.contains('Stress', na=False)] if hasattr(errors.index, 'str') else pd.DataFrame()\n",
                "\n",
                "html_output = \"\"\n",
                "if len(energy_data) > 0:\n",
                "    html_output += create_booktabs_table(energy_data, 'Energy')\n",
                "if len(force_data) > 0:\n",
                "    html_output += create_booktabs_table(force_data, 'Force')\n",
                "if len(stress_data) > 0:\n",
                "    html_output += create_booktabs_table(stress_data, 'Stress')\n",
                "\n",
                "# If no property separation worked, use all data\n",
                "if not html_output:\n",
                "    html_output = create_booktabs_table(errors, 'Validation')\n",
                "\n",
                "display(HTML(html_output))"
            ]
            
            return code_lines

        def _dataframe_to_code(self, df):
            """
            Convert DataFrame to Python code that recreates it.
            """
            import pandas as pd
            
            # Get the data as dict
            data_dict = df.to_dict('index')
            
            # Convert to string representation
            df_code = f"pd.DataFrame.from_dict({repr(data_dict)}, orient='index')"
            
            # If it has a MultiIndex, we need to recreate that
            if hasattr(df.index, 'names') and len(df.index.names) > 1:
                index_tuples = [tuple(idx) if hasattr(idx, '__len__') else (idx,) for idx in df.index]
                df_code += f"\nlines[-1].index = pd.MultiIndex.from_tuples({repr(index_tuples)}, names={repr(df.index.names)})"
                df_code = f"(lambda: (lines := [{df_code}]) and lines[0])()"  # Ugly but works
            
            return df_code

        def _execute_code(self, code_lines):
            """
            Execute the code and capture HTML output.
            """
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            # Join code lines into a single string
            code = ''.join(code_lines)
            
            # Create execution environment
            exec_globals = {}
            exec_locals = {}
            
            # Capture output
            captured_html = []
            
            # Mock display function to capture HTML
            def mock_display(html_obj):
                if hasattr(html_obj, 'data'):
                    captured_html.append(html_obj.data)
                elif hasattr(html_obj, '_repr_html_'):
                    captured_html.append(html_obj._repr_html_())
                else:
                    captured_html.append(str(html_obj))
            
            # Mock HTML class
            class MockHTML:
                def __init__(self, data):
                    self.data = data
                    captured_html.append(data)
            
            # Set up execution environment
            exec_globals['display'] = mock_display
            exec_globals['HTML'] = MockHTML
            
            try:
                # Execute the code
                exec(code, exec_globals, exec_locals)
                
                # Return captured HTML
                if captured_html:
                    return captured_html[-1]  # Return the last HTML output
                else:
                    return "<p>No output generated</p>"
                    
            except Exception as e:
                return f"<p>Error executing code: {str(e)}</p>"


        def _generate_validation_tables(self, errors):
            """
            Generate formatted markdown tables from the errors DataFrame with booktabs styling.
            
            Args:
                errors: pandas DataFrame with metrics
                
            Returns:
                str: Markdown formatted tables
            """
            
            # Separate by property type
            energy_data = errors[errors.index.get_level_values(-1) == 'Energy'].copy() if hasattr(errors.index, 'get_level_values') else errors[errors['Property'] == 'Energy'].copy() if 'Property' in errors.columns else []
            force_data = errors[errors.index.get_level_values(-1) == 'Force'].copy() if hasattr(errors.index, 'get_level_values') else errors[errors['Property'] == 'Force'].copy() if 'Property' in errors.columns else []
            stress_data = errors[errors.index.get_level_values(-1) == 'Stress'].copy() if hasattr(errors.index, 'get_level_values') else errors[errors['Property'] == 'Stress'].copy() if 'Property' in errors.columns else []
            
            markdown = []
            
            if len(energy_data) > 0:
                markdown.append("## Energy Metrics")
                markdown.append("")
                markdown.append(self._format_property_table(energy_data, "Energy"))
                
            if len(force_data) > 0:
                markdown.append("## Force Metrics")
                markdown.append("")
                markdown.append(self._format_property_table(force_data, "Force"))
                
            if len(stress_data) > 0:
                markdown.append("## Stress Metrics")
                markdown.append("")
                markdown.append(self._format_property_table(stress_data, "Stress"))
            
            # If we couldn't separate by property, just format the whole thing
            if not markdown:
                markdown.append("## Validation Metrics")
                markdown.append("")
                markdown.append(self._format_general_table(errors))
            
            return "\n".join(markdown)
        
        def _format_property_table(self, data, property_name):
            """
            Format a property-specific table with clean booktabs-like styling.
            
            Args:
                data: pandas DataFrame for a specific property
                property_name: str, name of the property
                
            Returns:
                str: Markdown formatted table with booktabs styling
            """
            
            if len(data) == 0:
                return f"No {property_name.lower()} data available."
            
            # Separate data into groups and ALL rows
            all_rows = []
            group_rows = []
            
            for idx, row in data.iterrows():
                if hasattr(idx, '__len__') and len(idx) >= 3:  # Multi-index
                    group = idx[0]
                    weight_type = idx[1]
                    train_test = idx[2]
                else:
                    # Try to extract from row data if available
                    group = getattr(row, 'Group', str(idx))
                    weight_type = getattr(row, 'Weight', 'Unknown')
                    train_test = getattr(row, 'Split', 'Unknown')
                
                row_data = {
                    'Group': group,
                    'Weight': weight_type,
                    'Split': train_test,
                    'Count': int(row.get('ncount', row.get('Count', 0))),
                    'MAE': row.get('mae', row.get('MAE', 0)),
                    'RMSE': row.get('rmse', row.get('RMSE', 0)),
                    'R²': row.get('rsq', row.get('R2', 0))
                }
                
                if group == '*ALL':
                    all_rows.append(row_data)
                else:
                    group_rows.append(row_data)
            
            # Sort group rows for better presentation
            group_rows.sort(key=lambda x: (x['Group'], x['Weight'], x['Split']))
            all_rows.sort(key=lambda x: (x['Weight'], x['Split']))
            
            # Create separate tables for weighted and unweighted if we have both
            unweighted_groups = [row for row in group_rows if 'weighted' not in row['Weight'].lower()]
            weighted_groups = [row for row in group_rows if 'weighted' in row['Weight'].lower()]
            unweighted_all = [row for row in all_rows if 'weighted' not in row['Weight'].lower()]
            weighted_all = [row for row in all_rows if 'weighted' in row['Weight'].lower()]
            
            lines = []
            
            # Unweighted table
            if unweighted_groups or unweighted_all:
                lines.append(f"### {property_name} Metrics (Unweighted)")
                lines.append("")
                lines.append("| Group | Train/Test | Count | MAE | RMSE | R² |")
                lines.append("|-------|------------|-------|-----|------|----|") 
                
                # Add group rows first
                for row in unweighted_groups:
                    lines.append(f"| {row['Group']} | {row['Split']} | {row['Count']:,} | {row['MAE']:.6f} | {row['RMSE']:.6f} | {row['R²']:.6f} |")
                
                # Add separator and ALL rows at bottom
                if unweighted_groups and unweighted_all:
                    lines.append("|-------|------------|-------|-----|------|----|") 
                
                for row in unweighted_all:
                    lines.append(f"| **ALL** | **{row['Split']}** | **{row['Count']:,}** | **{row['MAE']:.6f}** | **{row['RMSE']:.6f}** | **{row['R²']:.6f}** |")
                
                lines.append("")
            
            # Weighted table
            if weighted_groups or weighted_all:
                lines.append(f"### {property_name} Metrics (Weighted)")
                lines.append("")
                lines.append("| Group | Train/Test | Count | MAE | RMSE | R² |")
                lines.append("|-------|------------|-------|-----|------|----|") 
                
                # Add group rows first
                for row in weighted_groups:
                    lines.append(f"| {row['Group']} | {row['Split']} | {row['Count']:,} | {row['MAE']:.6f} | {row['RMSE']:.6f} | {row['R²']:.6f} |")
                
                # Add separator and ALL rows at bottom
                if weighted_groups and weighted_all:
                    lines.append("|-------|------------|-------|-----|------|----|") 
                
                for row in weighted_all:
                    lines.append(f"| **ALL** | **{row['Split']}** | **{row['Count']:,}** | **{row['MAE']:.6f}** | **{row['RMSE']:.6f}** | **{row['R²']:.6f}** |")
                
                lines.append("")
            
            return "\n".join(lines)
        
        def _format_general_table(self, data):
            """
            Format a general table when property separation isn't possible, using booktabs styling.
            """
            # Group data by property and weight type
            all_rows = []
            group_rows = []
            
            for idx, row in data.iterrows():
                # Try to extract information from index or row
                if hasattr(idx, '__len__') and len(idx) >= 4:  # Multi-index with property
                    group = idx[0]
                    weight_type = idx[1]
                    train_test = idx[2]
                    prop = idx[3]
                elif hasattr(idx, '__len__') and len(idx) >= 3:
                    group = idx[0]
                    weight_type = idx[1]
                    train_test = idx[2]
                    prop = 'Unknown'
                else:
                    group = str(idx)
                    weight_type = 'Unknown'
                    train_test = 'Unknown'
                    prop = 'Unknown'
                
                row_data = {
                    'Group': group,
                    'Weight': weight_type,
                    'Split': train_test,
                    'Property': prop,
                    'Count': int(row.get('ncount', row.get('Count', 0))),
                    'MAE': row.get('mae', row.get('MAE', 0)),
                    'RMSE': row.get('rmse', row.get('RMSE', 0)),
                    'R²': row.get('rsq', row.get('R2', 0))
                }
                
                if group == '*ALL':
                    all_rows.append(row_data)
                else:
                    group_rows.append(row_data)
            
            # Sort rows for better presentation
            group_rows.sort(key=lambda x: (x['Property'], x['Group'], x['Weight'], x['Split']))
            all_rows.sort(key=lambda x: (x['Property'], x['Weight'], x['Split']))
            
            lines = []
            lines.append("| Group | Property | Train/Test | Count | MAE | RMSE | R² |")
            lines.append("|-------|----------|------------|-------|-----|------|----|") 
            
            # Add group rows first
            for row in group_rows:
                lines.append(f"| {row['Group']} | {row['Property']} | {row['Split']} | {row['Count']:,} | {row['MAE']:.6f} | {row['RMSE']:.6f} | {row['R²']:.6f} |")
            
            # Add separator and ALL rows at bottom
            if group_rows and all_rows:
                lines.append("|-------|----------|------------|-------|-----|------|----|") 
            
            for row in all_rows:
                lines.append(f"| **ALL** | **{row['Property']}** | **{row['Split']}** | **{row['Count']:,}** | **{row['MAE']:.6f}** | **{row['RMSE']:.6f}** | **{row['R²']:.6f}** |")
            
            return "\n".join(lines)


        def write_potential(self, coeffs):
            
            # Support both ACE and PYACE sections
            if "ACE" in self.config.sections:
                ace_section = self.config.sections["ACE"]
            elif "PYACE" in self.config.sections:
                ace_section = self.config.sections["PYACE"]
            else:
                raise RuntimeError("No ACE or PYACE section found for PACE output")

            self.bzeroflag = ace_section.bzeroflag 
            self.numtypes = ace_section.numtypes
            self.ranks = ace_section.ranks
            self.lmin = ace_section.lmin
            self.lmax = ace_section.lmax
            self.nmax = ace_section.nmax
            self.mumax = ace_section.mumax
            self.nmaxbase = ace_section.nmaxbase
            self.rcutfac = ace_section.rcutfac
            self.lmbda = ace_section.lmbda
            self.rcinner = ace_section.rcinner
            self.drcinner = ace_section.drcinner
            self.types = getattr(ace_section, 'types', getattr(ace_section, 'elements', ['H']))  # ACE uses 'types', PYACE uses 'elements'
            self.erefs = ace_section.erefs
            
            # Handle attributes that may not exist in PYACE
            self.bikflag = getattr(ace_section, 'bikflag', False)
            self.b_basis = getattr(ace_section, 'b_basis', 'pa_tabulated')
            self.wigner_flag = getattr(ace_section, 'wigner_flag', True)

            if self.bzeroflag:
                #assert len(self.types) ==  len(self.erefs), "must provide reference energy for each atom type"
                if len(self.types) ==  len(self.erefs):
                    reference_ens = [float(e0) for e0 in self.erefs]
                else:
                    reference_ens = [0.0] * len(self.types)
            elif not self.bzeroflag:
                reference_ens = [0.0] * len(self.types)
            bondinds=range(len(self.types))
            bonds = [b for b in itertools.product(bondinds,bondinds)]
            bondstrs = ['[%d, %d]' % b for b in bonds]
            assert len(self.lmbda) == len(bondstrs), "must provide rc, lambda, for each BOND type"
            assert len(self.rcutfac) == len(bondstrs), "must provide rc, lambda, for each BOND type"
            if len(self.lmbda) == 1:
                lmbdavals = self.lmbda
                rcvals = self.rcutfac
                rcinnervals = self.rcinner
                drcinnervals = self.drcinner
            if len(self.lmbda) > 1:
                lmbdavals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.lmbda)}
                rcvals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.rcutfac)}
                rcinnervals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.rcinner)}
                drcinnervals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.drcinner)}


            ldict = {int(rank):int(lmax) for rank,lmax in zip(self.ranks,self.lmax)}
            L_R = 0
            M_R = 0
            rankstrlst = ['%s']*len(self.ranks)
            rankstr = ''.join(rankstrlst) % tuple(self.ranks)
            lstrlst = ['%s']*len(self.ranks)
            lstr = ''.join(lstrlst) % tuple(self.lmax)
            if not self.wigner_flag:
                try:
                    with open('cg_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
                        ccs = pickle.load(handle)
                except FileNotFoundError:
                    ccs = get_cg_coupling(ldict,L_R=L_R)
                    #print (ccs)
                    #store them for later so they don't need to be recalculated
                    store_generalized(ccs, coupling_type='cg',L_R=L_R)
            else:
                try:
                    with open('wig_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
                        ccs = pickle.load(handle)
                except FileNotFoundError:
                    ccs = get_wig_coupling(ldict,L_R)
                    #print (ccs)
                    #store them for later so they don't need to be recalculated
                    store_generalized(ccs, coupling_type='wig',L_R=L_R)


            apot = AcePot(self.types, reference_ens, [int(k) for k in self.ranks], [int(k) for k in self.nmax],  [int(k) for k in self.lmax], self.nmaxbase, rcvals, lmbdavals, rcinnervals, drcinnervals, [int(k) for k in self.lmin], self.b_basis, **{'ccs':ccs[M_R]})
            if ace_section.bzeroflag:
                apot.set_betas(coeffs,has_zeros=False)
            else:
                apot.set_betas(coeffs,has_zeros=True)
            
            # Handle nus attribute - may not exist in PYACE
            if hasattr(ace_section, 'nus'):
                apot.set_funcs(nulst=ace_section.nus)
            
            apot.write_pot(self.config.sections["OUTFILE"].potential_name)
            # Append metadata to .yace file
            unit = f"# units {self.config.sections['REFERENCE'].units}\n"
            atom = f"# atom_style {self.config.sections['REFERENCE'].atom_style}\n"
            pair = "\n".join(["# " + s for s in self.config.sections["REFERENCE"].lmp_pairdecl]) + "\n"
            refsec = unit + atom + pair
            with open(f"{self.config.sections['OUTFILE'].potential_name}.yace", "a") as fp:
                fp.write("# This file was generated by FitSNAP.\n")
                fp.write(f"# Hash: {self.config.hash}\n")
                fp.write(f"# FitSNAP REFERENCE section settings:\n")
                fp.write(f"{refsec}")

except ModuleNotFoundError:

    class Pace(Output):
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("Missing sympy or pyyaml modules.")

def _to_coeff_string(coeffs, config):
    """
    Convert a set of coefficients along with descriptor options to a coeffs file.
    """

    # Support both ACE and PYACE sections
    if "ACE" in config.sections:
        desc_str = "ACE"
        ace_section = config.sections["ACE"]
    elif "PYACE" in config.sections:
        desc_str = "PYACE"
        ace_section = config.sections["PYACE"]
    else:
        raise RuntimeError("No ACE or PYACE section found for coefficient output")
    
    # Check if we have the required attributes for coefficient output
    if not hasattr(ace_section, 'blank2J') or not hasattr(ace_section, 'blist'):
        raise RuntimeError(f"Section {desc_str} missing required attributes 'blank2J' or 'blist' for coefficient output. PYACE coefficient output may not be supported.")
    
    coeffs = coeffs.reshape((ace_section.numtypes, -1))
    blank2Js = ace_section.blank2J.reshape((ace_section.numtypes, -1))
    if ace_section.bzeroflag:
        coeff_names = ace_section.blist
    else:
        coeff_names = [[0]]+ace_section.blist
    coeffs = np.multiply(coeffs, blank2Js)
    type_names = getattr(ace_section, 'types', getattr(ace_section, 'elements', ['H']))
    out = f"# FitSNAP generated on {datetime.now()} with Hash: {config.hash}\n\n"
    out += "{} {}\n".format(len(type_names), int(np.ceil(len(coeff_names)/ace_section.numtypes)))
    for elname, column in zip(type_names,
                        coeffs):
        out += "{}\n".format(elname)
        out += "\n".join(f" {bval:<30.18} #  B{bname} " for bval, bname in zip(column, coeff_names))
        out += "\n"
    out += "\n# End of potential"
    return out

def _to_potential_file(config):
    """
    Use config settings to write a LAMMPS potential .mod file.
    """
    
    # Support both ACE and PYACE sections
    if "ACE" in config.sections:
        ace_section = config.sections["ACE"]
    elif "PYACE" in config.sections:
        ace_section = config.sections["PYACE"]
    else:
        raise RuntimeError("No ACE or PYACE section found for potential file output")

    ps = config.sections["REFERENCE"].lmp_pairdecl[0]
    ace_filename = config.sections["OUTFILE"].potential_name.split("/")[-1]

    out = "# This file was generated by FitSNAP.\n"
    out += f"# Hash: {config.hash}\n\n"

    if "hybrid" in ps:
        # extract non zero parts of pair style
        if "zero" in ps.split():
            split = ps.split()
            zero_indx = split.index("zero")
            del split[zero_indx]
            del split[zero_indx] # delete the zero pair cutoff
            ps = ' '.join(split)
        out += ps + " pace product\n"
        # add pair coeff commands from input, ignore if pair zero
        for pc in config.sections["REFERENCE"].lmp_pairdecl[1:]:
            out += f"{pc}\n" if "zero" not in pc else ""
        pc_ace = f"pair_coeff * * pace {ace_filename}.yace"
        type_names = getattr(ace_section, 'types', getattr(ace_section, 'elements', ['H']))
        for t in type_names:
            pc_ace += f" {t}"
        out += pc_ace
    else:
        out += "pair_style pace product\n"
        pc_ace = f"pair_coeff * * {ace_filename}.yace" 
        type_names = getattr(ace_section, 'types', getattr(ace_section, 'elements', ['H']))
        for t in type_names:
            pc_ace += f" {t}"
        out += pc_ace

    return out

