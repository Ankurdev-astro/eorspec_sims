import os

dir_path = "gen_schedules"

for fname in os.listdir(dir_path):
    if fname.endswith(".txt"):
        fpath = os.path.join(dir_path, fname)
        with open(fpath) as f:
            lines = [line for line in f if line.strip() and not 
                     (line.startswith("#") or line.startswith("Cerro-Chajnantor"))]
            if 0 < len(lines) < 10:
                print(f"WARNING: {fname} has only {len(lines)} entries.")
            if len(lines) == 0:
                print(f"WARNING: Removing {fname} !")
                os.remove(fpath)         
