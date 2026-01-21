import pandas as pd
import matplotlib.pyplot as plt
import uproot
# sWeighted data
data = pd.read_hdf("/ceph/submit/data/user/a/anbeck/B2KPiMM_michele/sweights/standard/data_qsq-1.1-7.0/0.h5")

DATADIR = "/ceph/submit/data/user/a/anbeck/B2KPiMM_michele"

def load_root_dataframe(path):
    with uproot.open(path) as f:
        return f["B02KstMuMu_Run1_centralQ2E_sig"].arrays(library="pd")

references = {"wA0":  load_root_dataframe(f"{DATADIR}/A0.root"),"wApp": load_root_dataframe(f"{DATADIR}/A1.root"),"wS":   load_root_dataframe(f"{DATADIR}/AS.root"),}

variables = [("mKpi", r"$m(K\pi)$ [GeV/$c^2$]", (0.65,1.5)),("q2",   r"$q^2$ [GeV$^2/c^4$]", (1.1, 7.0)),]

weights = [("wA0",  r"$n_0^P$", "n0P"),("wApp", r"$n_1^P$", "n1P"),("wS",   r"$n_0^S$", "n0S"),("wAq",  r"$n_{\beta}$", "nBeta"),]

for var, xlabel, fixed_range in variables:

    if fixed_range is None:
        vmin, vmax = data[var].min(), data[var].max()
    else:
        vmin, vmax = fixed_range

    for wkey, ylabel, tag in weights:

        plt.figure(figsize=(6, 5))

        # sweighted histogram
        plt.hist(data[var],bins=80,range=(vmin, vmax),weights=data[wkey],histtype="stepfilled",linewidth=1,color="red",label="sWeighted data", density = True)

        #Reference 
        if wkey in references:
            ref = references[wkey]
            plt.hist(ref[var],bins=80,range=(vmin, vmax),histtype="step",linewidth= 3,color='green',label="Reference", density=True)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(fr"{xlabel} – {ylabel}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{var}_{tag}_sweighted_vs_ref.png", dpi=300)
        plt.close()



