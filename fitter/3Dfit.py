import numpy as np
import uproot
import zfit
import yaml
import pandas as pd
import mypdfs
import angularfunctions as af
import matplotlib.pyplot as plt
np.random.seed(0)
zfit.settings.set_seed(0)
zfit.settings.set_verbosity(5)


cosh = zfit.Space("cosh", limits=(-1, 1))
cosl = zfit.Space("cosl", limits=(-1, 1))
mass = zfit.Space("B_mass", limits=(5200, 5500))

angles = cosh * cosl
obs3D = angles * mass

#data loading

filename = "/home/submit/raynahp/IAP2026/data/full.root"
treename = "B02KstMuMu_Run1_centralQ2E_sig"



with uproot.open(filename) as f:
    df = f[treename].arrays(library="pd")


with uproot.open("../genbkg/generated_data.root") as f:
    df_bkg = f["background"].arrays(library="pd")

#fixed units
df_bkg["B_mass"] *= 1000.0


# selections
df = df[
    df["q2"].between(1.1, 7.0) &
    df["mKpi"].between(0.65, 1.5) &
    (df["B_mass"].between(5200, 5500))
]

# Original signal-only data (before adding df_bkg)
df_signal_only = df.copy()

df_signal_only["cosh"] = df_signal_only["cosThetaK"]
df_signal_only["cosl"] = df_signal_only["cosThetaL"]

df = pd.concat([df, df_bkg], ignore_index=True)

#plot individual contributions A0 ect ect copy bottom bit of angular distribution and sweighted mkpi and q^2
df["cosh"] = df["cosThetaK"]
df["cosl"] = df["cosThetaL"]

data = zfit.Data.from_pandas(
    df[["cosh", "cosl", "B_mass"]],
    obs=obs3D
)


# total yield

#different components of pdf before we do the fit in B Mass and the 2 anlges including the data 
#if the total pdf is very different from the data try to figure out why. 
#push to git 
#send error output
#try fixing other parameters from 1d fit
#nsig are a little low 



App = zfit.Parameter("App", 0.1670, -1.0, 2.0)
A0 = zfit.Parameter("A0", 0.5, -1.0, 2.0)
Aqs = zfit.Parameter("Aqs", 0.01, -10.0, 10.0)
Aqc = zfit.Parameter("Aqc", 0.01, -10.0, 10.0)
AfbHS = zfit.Parameter("AfbHS", 0.0, -1.0, 1.0)
AfbHC = zfit.Parameter("AfbHC", 0.0, -1.0, 1.0)
AfbLS = zfit.Parameter("AfbLS", 0.0, -1.0, 1.0)
AfbLC = zfit.Parameter("AfbLC", 0.0, -1.0, 1.0)

def ASconditions(params):
    # The sum of all amplitudes must be 1.
    # This means that AS is not a free parameter.
    return 1-params['A0']-params['App']-params['Aqc']-params['Aqs']



AS = zfit.ComposedParameter("AS", ASconditions,
                            params={'A0': A0, 'App': App, 'Aqc': Aqc, 'Aqs': Aqs})




pdf_sig_ang = mypdfs.my2Dpdf(obs=angles,App=App, A0=A0, AS=AS,Aqc=Aqc, Aqs=Aqs,AfbHC=AfbHC, AfbHS=AfbHS,AfbLC=AfbLC, AfbLS=AfbLS,)

#signal mass PDF 

mu = zfit.Parameter("mu", 5280, 5200, 5400)
#fixed all these values
sigmal = zfit.Parameter("sigmal", 15.9, 5, 80)
alphal = zfit.Parameter("alphal", 1.36, 0.1, 5)
nl     = zfit.Parameter("nl", 9.77, 0.5, 50)

sigmar = zfit.Parameter("sigmar", 15.5, 5, 80)
alphar = zfit.Parameter("alphar", 1.66, 0.1, 5)
nr     = zfit.Parameter("nr", 146, 0.5, 500)

for p in [sigmal, alphal, nl, sigmar, alphar, nr]:
    p.floating = False

#fix to value that I got from 1D fit everything but mu 
pdf_sig_mass = zfit.pdf.GeneralizedCB(
    mu, sigmal, alphal, nl,
    sigmar, alphar, nr,
    obs=mass
)

#signal 3D PDF

pdf_sig_3D = zfit.pdf.ProductPDF([
    pdf_sig_ang,
    pdf_sig_mass
])


#angular background PDF
a1_cosh = zfit.Parameter("a1_cosh", 0.0, -1, 1)
a2_cosh = zfit.Parameter("a2_cosh", -0.2, -1, 1)
a1_cosl = zfit.Parameter("a1_cosl", 0.0, -1, 1)
a2_cosl = zfit.Parameter("a2_cosl", -0.4, -1, 1)



for p in [a1_cosh, a2_cosh, a1_cosl, a2_cosl]:
    p.floating = False

pdf_bkg_cosh = zfit.pdf.Legendre(obs=cosh, coeffs=[a1_cosh, a2_cosh])
pdf_bkg_cosl = zfit.pdf.Legendre(obs=cosl, coeffs=[a1_cosl, a2_cosl])




pdf_bkg_ang = zfit.pdf.ProductPDF(
    [pdf_bkg_cosh, pdf_bkg_cosl],
    obs=angles
)

#mass background PDF
lam = zfit.Parameter("lambda", -0.001, -1, 0)
pdf_bkg_mass = zfit.pdf.Exponential(obs=mass, lam=lam)

#background 3D PDF

pdf_bkg_3D = zfit.pdf.ProductPDF([
    pdf_bkg_ang,
    pdf_bkg_mass
])

#extend everything

Nsig = zfit.Parameter("Nsig", 2000000, 0, 1e8)
Nbkg = zfit.Parameter("Nbkg", 5000000, 0, 1e8)

pdf_sig_ext = pdf_sig_3D.create_extended(Nsig)
pdf_bkg_ext = pdf_bkg_3D.create_extended(Nbkg)


#total model
model = zfit.pdf.SumPDF([
    pdf_sig_ext,
    pdf_bkg_ext
])


vals_sig = pdf_sig_3D.pdf(data).numpy()
vals_bkg = pdf_bkg_3D.pdf(data).numpy()

print("Signal PDF min:", np.min(vals_sig), "NaNs:", np.isnan(np.log((vals_sig))).sum())
print("Bkg PDF min:", np.min(vals_bkg), "NaNs:", np.isnan(np.log((vals_bkg))).sum())




#fit everything

loss = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
minimizer = zfit.minimize.Minuit()
print (loss)
result = minimizer.minimize(loss)
print(result)
result.errors()
print(result)

#plotting


n_bins = 60
mass_min, mass_max = 5200, 5500

bins = np.linspace(mass_min, mass_max, n_bins + 1)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
bin_width = bins[1] - bins[0]


pdf_sig_mass_proj = pdf_sig_ext.create_projection_pdf(obs=mass)
pdf_bkg_mass_proj = pdf_bkg_ext.create_projection_pdf(obs=mass)
pdf_tot_mass_proj = model.create_projection_pdf(obs=mass)


x_data = zfit.Data.from_numpy(mass, bin_centers)

# Evaluate mass PDFs
y_sig = pdf_sig_mass_proj.ext_pdf(x_data).numpy() * bin_width
y_bkg = pdf_bkg_mass_proj.ext_pdf(x_data).numpy() * bin_width
y_tot = pdf_tot_mass_proj.ext_pdf(x_data).numpy() * bin_width



plt.figure(figsize=(8,6))



plt.hist(df["B_mass"], bins=bins, range=(5200, 5500),
         histtype="step", color="black", label="Data")
plt.hist(df_signal_only["B_mass"], bins=bins, range=(5200, 5500),
         histtype="step", color="green", label="Reference")
plt.plot(bin_centers, y_tot, label="Total fit", color="black")
plt.plot(bin_centers, y_sig, "--", label="Signal", color="red")
plt.plot(bin_centers, y_bkg, "--", label="Background", color="blue")
plt.xlabel("B Mass [MeV]")
plt.ylabel("Events / bin")
plt.legend()
plt.savefig("angularfit3D_massfit.png")
plt.close()


#angle plots
n_bins_ang = 50
cosh_bins = np.linspace(-1, 1, n_bins_ang + 1)
cosl_bins = np.linspace(-1, 1, n_bins_ang + 1)

cosh_centers = 0.5 * (cosh_bins[1:] + cosh_bins[:-1])
cosl_centers = 0.5 * (cosl_bins[1:] + cosl_bins[:-1])

bin_width_cosh = cosh_bins[1] - cosh_bins[0]
bin_width_cosl = cosl_bins[1] - cosl_bins[0]

#1d projections
pdf_sig_cosh_proj = pdf_sig_ext.create_projection_pdf(obs=cosh)
pdf_bkg_cosh_proj = pdf_bkg_ext.create_projection_pdf(obs=cosh)
pdf_tot_cosh_proj = model.create_projection_pdf(obs=cosh)

pdf_sig_cosl_proj = pdf_sig_ext.create_projection_pdf(obs=cosl)
pdf_bkg_cosl_proj = pdf_bkg_ext.create_projection_pdf(obs=cosl)
pdf_tot_cosl_proj = model.create_projection_pdf(obs=cosl)

#
x_cosh_data = zfit.Data.from_numpy(array=cosh_centers, obs=cosh)
x_cosl_data = zfit.Data.from_numpy(array=cosl_centers, obs=cosl)

# Evaluate PDFs
y_sig_cosh = pdf_sig_cosh_proj.ext_pdf(x_cosh_data).numpy() * bin_width_cosh
y_bkg_cosh = pdf_bkg_cosh_proj.ext_pdf(x_cosh_data).numpy() * bin_width_cosh
y_tot_cosh = pdf_tot_cosh_proj.ext_pdf(x_cosh_data).numpy() * bin_width_cosh

y_sig_cosl = pdf_sig_cosl_proj.ext_pdf(x_cosl_data).numpy() * bin_width_cosl
y_bkg_cosl = pdf_bkg_cosl_proj.ext_pdf(x_cosl_data).numpy() * bin_width_cosl
y_tot_cosl = pdf_tot_cosl_proj.ext_pdf(x_cosl_data).numpy() * bin_width_cosl

#plot cosh
plt.figure(figsize=(8,6))
plt.hist(df["cosh"], bins=cosh_bins, range=(-1, 1),
         histtype="step", color="black", label="Data")
plt.hist(df_signal_only["cosh"], bins=cosh_bins, range=(-1, 1),
         histtype="step", color="green", label="Reference")
plt.plot(cosh_centers, y_tot_cosh, label="Total fit", color="black")
plt.plot(cosh_centers, y_sig_cosh, "--", label="Signal", color="red")
plt.plot(cosh_centers, y_bkg_cosh, "--", label="Background", color="blue")
plt.xlabel("cos(theta_K)")
plt.ylabel("Events / bin")
plt.legend()
plt.tight_layout()
plt.savefig("angularfit3D_coshfit.png")
plt.close()

#plot cosl
plt.figure(figsize=(8,6))
plt.hist(df["cosl"], bins=cosl_bins, range=(-1, 1),
         histtype="step", color="black", label="Data")
plt.hist(df_signal_only["cosl"], bins=cosl_bins, range=(-1, 1),
         histtype="step", color="green", label="Reference")
plt.plot(cosl_centers, y_tot_cosl, label="Total fit", color="black")
plt.plot(cosl_centers, y_sig_cosl, "--", label="Signal", color="red")
plt.plot(cosl_centers, y_bkg_cosl, "--", label="Background", color="blue")
plt.xlabel("cos(theta_L)")
plt.ylabel("Events / bin")
plt.legend()
plt.tight_layout()
plt.savefig("angularfit3D_coslfit.png")
plt.close()








#results
out = {}
for p in result.params:
    out[p.name] = {
        "value": float(result.params[p]["value"]),
    }

with open("angularfit3D_results.yml", "w") as f:
    yaml.dump(out, f)
