import pandas as pd ## pd version 2.2.3
df = pd.read_csv("data.csv")
df = df.dropna()
df = df[~df['OBJECT_TYPE'].isin(['AV', 'AGENT'])]
df = df.groupby('TRACK_ID').filter(lambda x: len(x) > 10)
df = df.groupby('TRACK_ID').filter(lambda x: (x['X'].max() - x['X'].min() > 1) or 
                                   (x['Y'].max() - x['Y'].min() > 1))
df = df.sort_values(by=['TRACK_ID', 'TIMESTAMP'])
df.to_csv("result.csv", index=False)
