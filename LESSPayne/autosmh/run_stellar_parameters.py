import numpy as np
import sys, os, time
import yaml
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d

from LESSPayne.smh import Session
from LESSPayne.specutils import Spectrum1D 
from LESSPayne.smh.spectral_models import ProfileFittingModel, SpectralSynthesisModel
from LESSPayne.smh.photospheres.abundances import asplund_2009 as solar_composition
from LESSPayne.PayneEchelle.spectral_model import DefaultPayneModel

def run_stellar_parameters(cfg):
    name = cfg["output_name"]
    NNpath = cfg["NN_file"]
    outdir = cfg["output_directory"]
    figdir = cfg["figure_directory"]
    if not os.path.exists(outdir): os.makedirs(outdir)
    if not os.path.exists(figdir): os.makedirs(figdir)
    print("Saving to output directory:",outdir)
    print("Saving figures to output directory:",figdir)
    
    popt_fname = os.path.join(outdir, cfg["payne_fname"])
    smh_fname = os.path.join(outdir, cfg["smh_fname"])
    
    spcfg = cfg["run_stellar_parameters"]
    #print(spcfg)
    if spcfg["method"] not in ["rpa_calibration","manual_all"]:
        raise ValueError(
            f"run_stellar_parameters method={spcfg['method']} is not valid\n(rpa_calibration, manual_all)"
        )
    
    startall = time.time()
    
    ## Load results of normalization
    session = Session.load(smh_fname)
    
    ## Step 7: initialize stellar parameters
    if spcfg["method"] == "rpa_calibration":
        model = DefaultPayneModel.load(NNpath, 1)
        with np.load(popt_fname) as tmp:
            popt_best = tmp["popt_best"].copy()
            popt_print = tmp["popt_print"].copy()
        Teff, logg, MH, aFe = round(popt_print[0]), round(popt_print[1],2), round(popt_print[2],2), round(popt_print[3],2)
        outstr1 = f"run_stellar_parameters:\n  PayneEchelle: T/g/v/M/a = {Teff}/{logg}/1.00/{MH}/{aFe}"
        
        ## Corrections from empirical fit to RPA duplicates
        dT = -.0732691466 * Teff + 247.57
        dg = 8.11486e-5 * logg - 0.28526
        dM = -0.06242672*MH - 0.3167661
        Teff, logg, MH = int(Teff - dT), round(logg - dg,2), round(MH - dM, 2)
        
        ## TODO
        #if aFe > 0.2: aFe = 0.4
        #else: aFe = 0.0
        aFe = 0.4
        
        ## TODO offer different vt methods
        #vt = round(2.13 - 0.23 * logg,2) # kirby09
        vt = round(0.060 * logg**2 - 0.569*logg + 2.585, 2) # RPA duplicates

        outstr2 = f"  Calibrated = {Teff}/{logg}/{vt}/{MH}/{aFe}"
        session.add_to_notes(outstr1+"\n"+outstr2)

    elif spcfg["method"] == "manual_all":
        Teff = logg = vt = MH = aFe = None
        
    ## Override manual
    if spcfg["manual_Teff"] is not None:
        Teff = int(spcfg["manual_Teff"])
        print(f"Setting Teff={Teff}")
    if spcfg["manual_logg"] is not None:
        logg = round(float(spcfg["manual_logg"]),2)
        print(f"Setting logg={logg}")
    if spcfg["manual_vt"] is not None:
        vt = round(float(spcfg["manual_vt"]),2)
        print(f"Setting vt={vt}")
    if spcfg["manual_MH"] is not None:
        MH = round(float(spcfg["manual_MH"]),2)
        print(f"Setting MH={MH}")
    if spcfg["manual_aFe"] is not None:
        aFe = round(float(spcfg["manual_aFe"]),2)
        print(f"Setting aFe={aFe}")
    
    if Teff is None: raise ValueError("Teff is not specified")
    if logg is None: raise ValueError("logg is not specified")
    if vt   is None: raise ValueError("vt is not specified")
    if MH   is None: raise ValueError("MH is not specified")
    if aFe  is None: raise ValueError("aFe is not specified")
    
    print(f"Final stellar parameters: T/g/v/M/a = {Teff:.0f}/{logg:.2f}/{vt:.2f}/{MH:.2f}/{aFe:.2f}")
    session.set_stellar_parameters(Teff, logg, vt, MH, aFe)
    
    if spcfg["measure_eqw_abundances"]:
        print("Measuring EQW abundances")
        session.measure_abundances() # eqw

    ## Save
    session.save(smh_fname, overwrite=True)
    print(f"Total time run_stellar_parameters: {time.time()-startall:.1f}")

