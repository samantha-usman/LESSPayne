from optparse import OptionParser
import yaml
import os, sys, time

from LESSPayne.PayneEchelle.run_payne_echelle import run_payne_echelle
from LESSPayne.autosmh.run_normalization import run_normalization

if __name__=="__main__":
    start = time.time()
    
    parser = OptionParser()
    # Flags indicating which parts of the pipeline to run
    parser.add_option("-a", "--all", dest="run_all", action="store_true", default=False)
    parser.add_option("-1", "--payne", dest="run_payneechelle", action="store_true", default=False)
    parser.add_option("-2", "--norm", dest="run_normalization", action="store_true", default=False)
    parser.add_option("-3", "--eqw", dest="run_equivalent_width", action="store_true", default=False)
    parser.add_option("-4", "--synth", dest="run_synthesis", action="store_true", default=False)
    parser.add_option("-5", "--errors", dest="run_errors", action="store_true", default=False)
    
    # Manually set stellar parameters - this needs to be moved to the config file!!!
    parser.add_option("-T", "--Teff", dest="Teff",
                      help="Override Payne Teff", type="int")
    parser.add_option("-G", "--logg", dest="logg",
                      help="Override Payne logg", type="float")
    parser.add_option("-V", "--vt", dest="vt",
                      help="Override Payne vt", type="float")
    parser.add_option("-M", "--MH", dest="MH",
                      help="Override Payne MH", type="float")
    parser.add_option("-A", "--aFe", dest="aFe",
                      help="Override Payne aFe (default 0.4)", type="float")

    (options, args) = parser.parse_args()
    
    if options.run_all:
        print("Running all phases of LESSPayne")
        options.run_payneechelle = True
        options.run_normalization = True
        options.run_equivalent_width = True
        options.run_synthesis = True
        options.run_errors = True
    
    cfg_file = args[0]
    print(cfg_file)
    if not os.path.exists(cfg_file):
        sys.exit(f"input file '{cfg_file}' does not exist")
    with open(cfg_file, "rb") as fp:
        cfg = yaml.load(fp, yaml.FullLoader)
    print(cfg)
    ## TODO: we should write out a CFG file with the defaults filled in

    outdir = cfg["output_directory"]
    figdir = cfg["figure_directory"]
    if not os.path.exists(outdir):
        print("Creating output directory:",outdir)
        os.makedirs(outdir)
    if not os.path.exists(figdir):
        print("Creating figure directory:",figdir)
        os.makedirs(figdir)
    print("Saving to output directory:",outdir)
    print("Saving figures to output directory:",figdir)
    
    
    ## PayneEchelle
    # Input: config file, spectrum files to analyze
    # Output: payneechelle fit npz file
    # Optional output: figure for each order showing fit
    if options.run_payneechelle:
        run_payne_echelle(cfg)
    
    ## Normalization
    # Input: payneechelle fit, spectrum files to analyze
    # Output: SMHR file with normalization done, mask file
    # Optional output: order normalization stamps
    if options.run_normalization:
        run_normalization(cfg)
    
    ## EQW
    # Input: SMHR file with normalization done, mask file
    # Output: SMHR file with EQWs fit
    # Optional output: EQW stamp plot
    # Optional output: line abundance table
    if options.run_equivalent_width:
        run_equivalent_width(cfg)
    
    
    ## Synth
    # Input: SMHR file with normalization and eqw done
    # Output: SMHR file with syntheses fit
    # Optional output: synth stamp plot
    # Optional output: line abundance table
    
    ## Errors
    # Input: SMHR file with all abundances you want done
    # Optional input: SP covariance matrix
    # Output: SMHR file with errors propagated
    # Optional output: 

    print(f"Time to run LESSPayne: {time.time()-start}:.1f}")
    
