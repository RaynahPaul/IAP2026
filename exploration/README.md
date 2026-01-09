# Investigation of the data
We are considering a toy model of $B^0\to K\pi\mu\mu$ located here:
```
/ceph/submit/data/user/m/matzeni/AmplitudeDeltaCi-fullRun2/Software/toys/vesta_toys/rare_mode_toys/fit_results_toys/toy_gen_toys_genSM_genNoHad_gent0at4_MuOnly_GKvDFF_fitC9C10_fixAllNuisance_z0_noBR_noBkg_Swave_noAcc_withsmallMassRangeE_largestKpi_largerandlowerqsq_10M.root
```
It is convenient to define filepaths like
```
export DATADIR=/ceph/submit/data/user/a/anbeck/B2KPiMM_michele
```
to avoid typing out the full path everytime which makes the commands huge. The location can be accessed using $DATADIR. The `export` command has to be run every time a new terminal is opened.

The `plotter.py` script looks at the different distributions and data inside this file. It can be run using
```bash
export DATADIR=/ceph/submit/data/user/a/anbeck/B2KPiMM_michele
python plotter.py --data $DATADIR/full.root
```