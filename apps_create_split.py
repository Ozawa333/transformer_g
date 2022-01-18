import json
import os
import pathlib

def create_split(split="test", name="test"):
    paths = []
    roots = sorted(os.listdir(split))
    for folder in roots:
        root_path = os.path.join(split, folder)
        paths.append(root_path)


    with open(name+".json", "w") as f:
        json.dump(paths, f)
    
    return paths

# insert path to train and test
# path should be relative to root directory or absolute paths
paths_to_probs = ["./datasets/apps5000/train", "./datasets/apps5000/test"]
names = ["train", "test"]

all_paths = []
for index in range(len(paths_to_probs)):
    all_paths.extend(create_split(split=paths_to_probs[index], name=names[index]))

with open("train_and_test.json", "w") as f:
    print(f"Writing all paths. Length = {len(all_paths)}")
    json.dump(all_paths, f)
