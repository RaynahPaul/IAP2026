import argparse
import matplotlib.pyplot as plt
import uproot

parser = argparse.ArgumentParser(description="Plot histograms from ROOT file.")
parser.add_argument("--data", type=str, help="Path to the input ROOT file.")
args = parser.parse_args()

# Open the ROOT file
with uproot.open(args.data) as file:
    data = file["B02KstMuMu_Run1_centralQ2E_sig"].arrays(library="pd")

# Task 1) Inspect the data and identify the different variables. What are their units?
# Task 2) Create one plot per variable with five different distributions, each represented as a histogram:
#           a) the total distribution
#           b,c) the distribution when selecting q2 smaller or larger than 2
#           d,e) the distribution when selecting mKpi smaller or larger than 1.1
# Task 3) Do your figures have all the required features? (E.g. axis labels, legend, easily distinguishable colors or linestyles, etc.)
# Task 4) Can you interpret the differences between the distributions a-e?
