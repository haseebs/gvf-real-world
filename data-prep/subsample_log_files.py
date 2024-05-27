from tqdm import tqdm
import pandas as pd
from pathlib import Path
from glob import glob

path = "../data-logs/7month-logs-startnov"
SAVE_PATH = "../data-logs/7month-logs-subsampled"
DATA_PATHS = [Path(x) for x in glob(path.__str__() + "/*.log.gz")]
DATA_PATHS.sort()

t = tqdm(total=len(DATA_PATHS))
    
for i, logpath in tqdm(enumerate(DATA_PATHS)):
    #print(f"Processing {i+1}/{len(DATA_PATHS)}")
    df = pd.read_csv(logpath)
    df = df.iloc[::10]
    df.to_csv(SAVE_PATH + "/" + str(logpath).split("/")[-1])
    t.update(1)
