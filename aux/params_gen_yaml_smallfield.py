import argparse
import yaml
import subprocess

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate YAML configuration file for Small Field Scans",
        epilog="The schedule out-filename is based on patch name and first day of schedule")
parser.add_argument("start", help="Start time of Schedule config file")
parser.add_argument("stop", help="Stop time of Schedule config file")
parser.add_argument("field_name", help="Name of the patch; Format: FieldName")
parser.add_argument("--params_yaml", default="params.yaml", help="Output YAML file name")

# Parse the command line arguments
args = parser.parse_args()

#patch_name format: FieldName_MM_DD
field_name = args.field_name.upper()
# mm_dd = (args.start).split(' ')[0][-5:].replace('-','_')
mm_dd = (args.start).split(' ')[0][:].replace('-','_')
patch_name = '_'.join([field_name, mm_dd])
print(patch_name)
out_yaml = f"schedule_{patch_name}.txt"

#patch = HorizontalPatch(name,HORIZONTAL, weight, azmin, azmax, el, scantime)
#Rectangle :name, weight, RA_MIN, DEC_MAX, RA_MAX, DEC_MIN

#For Small Field centre and width scans format
#"--patch", f"{patch_name},1,+150.119,+2.205,2.5"
#"--patch", f"{patch_name},1,150,2,3.15" #Works 10.02.2024
config_items = \
["--site-lat", "-22.9855235",
"--site-lon", "-67.7403087",
"--site-alt", "5610",
"--site-name", "Cerro-Chajnantor",
"--telescope", "FYST",
"--patch-coord", "C",
"--start", args.start,
"--stop", args.stop,
"--patch", f"{patch_name},1,150,2,2.0",
"--el-min-deg", "40",
"--el-max-deg", "65",
"--sun-avoidance-angle-deg", "35",
"--ces-max-time-s", "300", #300
"--fp-radius-deg", "0.0",
"--gap-s", "0.0",
"--gap-small-s" , "0.0",
"--no-partial-scans",
"--out", f"./gen_schedules/{out_yaml}",]

# Write the configuration to a YAML file
with open(args.params_yaml, 'w') as yaml_file:
    yaml.dump(config_items, yaml_file, default_flow_style=False)

print(f"Config for writing schedule to: {out_yaml}")

# After writing the YAML file, call/run another python script
subprocess.run(["python", "make_schedule_files.py"])
