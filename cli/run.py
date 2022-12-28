from optparse import OptionParser
import yaml
import os, sys

if __name__=="__main__":
    parser = OptionParser()
    # Flags indicating which parts of the pipeline to run
    parser.add_option("-a", "--all", dest="run_all", action="store_true", default=False)
    (options, args) = parser.parse_args()
    print(options.run_all)
    
    cfg_file = args[0]
    if not os.path.exists(cfg_file):
        sys.exit(f"input file '{cfg_file}' does not exist")
    with open(cfg_file, "rb") as fp:
        cfg = yaml.load(fp, yaml.FullLoader)
    print(cfg)
    
    
