import numpy as np
import uproot
import zfit
import matplotlib.pyplot as plt
import yaml

np.random.seed(0)
zfit.settings.set_seed(0)
zfit.settings.set_verbosity(5)

filename = '/ceph/submit/data/user/a/anbeck/B2KPiMM_michele/full.root'
treename = "B02KstMuMu_Run1_centralQ2E_sig"
#open files 
#signal data
with uproot.open(filename) as f:
    df = f[treename].arrays(library="pd")

# apply signal-side cuts
df = df[
    df["q2"].between(1.1, 7.0) &
    df["mKpi"].between(0.65, 1.5)
]



#generated background
with uproot.open("../genbkg/generated_data.root") as f:
    df_bkg = f["background"].arrays(library="pd")

#fixed units
df_bkg["B_mass"] *= 1000.0

#same cuts on background
df_bkg = df_bkg[
    df_bkg["q2"].between(1.1, 7.0) &
    df_bkg["mKpi"].between(0.65, 1.5)
]



#combined the data
mass = zfit.Space("B_mass", limits=(5200, 5500))
combined_mass = np.concatenate([df["B_mass"], df_bkg["B_mass"]])
data = zfit.Data.from_numpy(obs=mass, array=combined_mass)


#generated crystal ball signal PDF
mu = zfit.Parameter("mu", 5280.0, 5200.0, 5400.0)

sigmal = zfit.Parameter("sigmal", 20.0, 5.0, 80.0)
alphal = zfit.Parameter("alphal", 1.5, 0.1, 5.0)
nl     = zfit.Parameter("nl", 1.0, 0.5, 50.0)

sigmar = zfit.Parameter("sigmar", 25.0, 5.0, 80.0)
alphar = zfit.Parameter("alphar", 2.0, 0.1, 5.0)
nr     = zfit.Parameter("nr", 1.0, 0.5, 50.0)

pdf_sig = zfit.pdf.GeneralizedCB(mu, sigmal, alphal, nl, sigmar, alphar, nr, obs=mass)

Nsig = zfit.Parameter("Nsig", 20000, 0.0, 1.0e8)
pdf_sig_ext = pdf_sig.create_extended(Nsig)

#extended background PDF: Exponential
lam = zfit.Parameter("lambda", -0.001, -1.0, 0.0)
pdf_bkg = zfit.pdf.Exponential(obs=mass, lam=lam)

Nbkg = zfit.Parameter("Nbkg", 50000, 0.0, 1.0e8)
pdf_bkg_ext = pdf_bkg.create_extended(Nbkg)


model = zfit.pdf.SumPDF([pdf_sig_ext, pdf_bkg_ext])

#checking for NaNs

vals_sig = pdf_sig.pdf(data).numpy()
vals_bkg = pdf_bkg.pdf(data).numpy()

print("Signal PDF check:")
print("  min =", np.min(vals_sig))
print("  NaNs =", np.isnan(vals_sig).sum())
print("  infs =", np.isinf(vals_sig).sum())

print("Background PDF check:")
print("  min =", np.min(vals_bkg))
print("  NaNs =", np.isnan(vals_bkg).sum())
print("  infs =", np.isinf(vals_bkg).sum())



print("Signal log-PDF NaNs:", np.isnan(np.log(vals_sig)).sum())
print("Background log-PDF NaNs:", np.isnan(np.log(vals_bkg)).sum())


#fit
loss = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
minimizer = zfit.minimize.Minuit()

result = minimizer.minimize(loss)
result.errors()

print(result)



out = {}
for p in result.params:
    out[p.name] = {
        "value": float(result.params[p]["value"]),
        "error_upper": float(result.params[p]["errors"]["upper"]),
        "error_lower": float(result.params[p]["errors"]["lower"]),
    }

with open("mass_fit_generalizedCB.yml", "w") as f:
    yaml.dump(out, f)


#plotting
nbins = 60
xmin, xmax = 5200, 5600

x = np.linspace(xmin, xmax, 600)
bin_width = (xmax - xmin) / nbins

plt.figure(figsize=(8, 6))

# Data histogram
plt.hist(combined_mass, bins=nbins, range=(xmin, xmax), histtype="step", color="black", label="Data")

# PDF curves
y_sig = pdf_sig_ext.ext_pdf(x).numpy() * bin_width
y_bkg = pdf_bkg_ext.ext_pdf(x).numpy() * bin_width
y_tot = model.ext_pdf(x).numpy() * bin_width

plt.plot(x, y_tot, color="black", label="Total fit")
plt.plot(x, y_sig, color="red", linestyle="--", label="Signal (CB)")
plt.plot(x, y_bkg, color="blue", linestyle="--", label="Background (Expo)")

plt.xlabel(r"$m(B)$ [MeV/$c^2$]")
plt.ylabel("Events / bin")
plt.legend()
plt.tight_layout()
plt.savefig("mass_fit_generalizedCB.pdf")
plt.close()
