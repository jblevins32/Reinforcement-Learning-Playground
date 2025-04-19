import pandas as pd
import os

for item in ['critic/','policy/','reward/']:
    file_path = '../data/' + item
    print(item)

    for filename in os.listdir(file_path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(file_path, filename))
            print(f'mean {round(df.Value[737:1000].mean(),3)} max {round(df.Value[737:1000].max(),3)} {filename}')