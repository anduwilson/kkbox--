import pandas as pd
from MeanEncoder import MeanEncoder

data = pd.read_csv("D:/BaiduNetdiskDownload/Music Recommendation/Data/train.csv/train.csv", usecols=["song_id", "target"])
data = data.head(1000)
encoder = MeanEncoder(['song_id'])
data = encoder.fit_transform(data, data["target"])
print(data)
