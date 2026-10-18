import os 
import json
from source.helpers import load_instance, PATH_TO_BNC_RESULTS, PATH_TO_INSTANCES, RS, KS, TIME_LIMIT
from tqdm import tqdm
from itertools import product
from source.bnc import solve_BNC
import numpy as np
    
if __name__ == "__main__":
    print(os.listdir(PATH_TO_INSTANCES))
    FILENAME_AND_EXTENSIONS = sorted(map(str, os.listdir(PATH_TO_INSTANCES)))
    for (filename_and_extension, r, k) in tqdm(product(FILENAME_AND_EXTENSIONS, RS, KS)):
        filename = os.path.splitext(filename_and_extension)[0] + f"_{r:.1f}_{k}" 
        if filename + ".json" in os.listdir(PATH_TO_BNC_RESULTS):# or optimum_already_known(filename_and_extension, r, k):
            continue

        p, s, t_max = load_instance(os.path.join(PATH_TO_INSTANCES, filename_and_extension))
        res = solve_BNC(p, s, t_max, r, K=k, time_limit=TIME_LIMIT)
        res["incumbent"] = res["incumbent"].tolist()
        with open(os.path.join(PATH_TO_BNC_RESULTS, filename + ".json"), "w+") as file:
            json.dump(res, file)


