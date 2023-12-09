import os

inf_dir = os.path.join("inference", "results")
valid_start_idxs = [x for x in range(0, 61, 4)]
for f in os.listdir(inf_dir):
    for sub_dir in os.listdir(os.path.join(inf_dir, f)):
        if sub_dir.__contains__("lively-meadow-695"):
            sub_sub_dir = os.listdir(sub_dir)[0]
            for file in os.listdir(os.path.join(inf_dir, f, sub_dir, sub_sub_dir, "labels")):
                print(file)
                start_idx = int(file.split("_")[2])
                if not valid_start_idxs.__contains__(start_idx):
                    print(f"Removing {file}")
                    print("in " + os.path.join(inf_dir, f, sub_dir, sub_sub_dir, "labels"))
                    # os.remove(os.path.join(inf_dir, f, sub_dir, sub_sub_dir, "labels", file))