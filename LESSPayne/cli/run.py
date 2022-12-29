from optparse import OptionParser
import yaml
import os, sys

if __name__=="__main__":
    parser = OptionParser()
    # Flags indicating which parts of the pipeline to run
    parser.add_option("-a", "--all", dest="run_all", action="store_true", default=False)
    parser.add_option("-p", "--payne", dest="run_payne4less", action="store_true", default=False)
    parser.add_option("-n", "--norm", dest="run_normalization", action="store_true", default=False)
    parser.add_option("-w", "--eqw", dest="run_equivalent_width", action="store_true", default=False)
    parser.add_option("-s", "--synth", dest="run_synthesis", action="store_true", default=False)
    parser.add_option("-e", "--errors", dest="run_errors", action="store_true", default=False)
    
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
        options.run_payne4less = True
        options.run_normalization = True
        options.run_equivalent_width = True
        options.run_synthesis = True
        options.run_errors = True
    
    cfg_file = args[0]
    if not os.path.exists(cfg_file):
        sys.exit(f"input file '{cfg_file}' does not exist")
    with open(cfg_file, "rb") as fp:
        cfg = yaml.load(fp, yaml.FullLoader)
    print(cfg)
    
    ## Payne4LESS
    # Input: config file, spectrum files to analyze
    # Output: payne4less fit npz file
    # Optional output: figure for each order showing fit
    
    ## Normalization
    # Input: payne4less fit, spectrum files to analyze
    # Output: SMHR file with normalization done, mask file
    # Optional output: order normalization stamps
    
    ## EQW
    # Input: SMHR file with normalization done, mask file
    # Output: SMHR file with EQWs fit
    # Optional output: EQW stamp plot
    # Optional output: line abundance table
    
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
