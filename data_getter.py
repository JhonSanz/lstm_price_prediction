import pandas as pd

class DataGetter:
    def __init__(self):
        self.data = None

    def get_initial_data(self, filename):
        self.data = pd.read_csv(filename, sep="\t")
        self.data['<DATE>'] = self.data['<DATE>'] + ' ' + self.data['<TIME>']
        self.data.rename(columns={
            "<DATE>": "date",
            "<OPEN>": "open",
            "<HIGH>": "high",
            "<LOW>": "low",
            "<CLOSE>": "close",
            "<SPREAD>": "spread"
        }, inplace=True)
        self.data = self.data[['date', 'open', 'high', 'low', 'close', 'spread']]
        self.data.reset_index(drop=True, inplace=True)
        self.data['date'] = pd.to_datetime(self.data['date'], format='%Y.%m.%d')
        self.data.loc[self.data["spread"] == 0, "spread"] = 1
        return self.data
