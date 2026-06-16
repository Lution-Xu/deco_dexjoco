import json
import pandas as pd

file = pd.read_parquet('~/libero/meta/tasks.parquet', engine='pyarrow')
out = {}
for i in range(len(file)):
    out[i] = file.iloc[i].name

with open('~/libero/meta/task_name.json', 'w') as f:
    json.dump(out, f)