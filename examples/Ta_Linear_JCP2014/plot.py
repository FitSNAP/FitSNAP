# import FitSNAP library tools for dataframe analysis

from fitsnap3lib.tools.dataframe_tools import DataframeTools

# make a dataframe tool object, and read the dataframe

dataframe_tool = DataframeTools("FitSNAP.df")
df = dataframe_tool.read_dataframe()

# plot energy agreement comparison as a line
# use peratom = False if total energy desired

#dataframe_tool.plot_agreement("Energy", fitting_set="Training", mode="Linear", peratom=False)

# plot energy agreement as a distribution
# use peratom = False if total energy desired

dataframe_tool.plot_agreement("Energy", fitting_set="Training", mode="Distribution", peratom=True)

dataframe_tool.plot_agreement("Force", fitting_set="Training", mode="Distribution")
