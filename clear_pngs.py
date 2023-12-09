import os

inf_dir = os.path.join("inference", "results")
valid_start_idxs = [x for x in range(0, 61, 4)]
for fragment_dir in os.listdir(inf_dir):
    print("Checking", fragment_dir)
    for run_dir in os.listdir(os.path.join(inf_dir, fragment_dir)):
        if run_dir.__contains__("lively-meadow-695"):
            print(run_dir, "contains lively-meadow-695")
            label_dir = [x for x in os.listdir(run_dir) if x.__contains__("labels")][0]
            label_sub_dir = os.listdir(os.path.join(inf_dir, fragment_dir, run_dir, label_dir))[0]
            for file in os.listdir(os.path.join(inf_dir, fragment_dir, run_dir, label_dir, label_sub_dir)):
                print(file)
                start_idx = int(file.split("_")[2])
                if not valid_start_idxs.__contains__(start_idx):
                    print(f"Removing {file}")
                    print("in " + os.path.join(inf_dir, fragment_dir, run_dir, label_dir, label_sub_dir))
                    # os.remove(os.path.join(inf_dir, f, sub_dir, sub_sub_dir, "labels", file))