import pandas as pd

class ProccessMetatraderData:
    def __init__(self):
        self.data = None

    def read_data(self):
        self.data = pd.read_csv(
            "zigzag_data.csv",
            names=[
                "date", "time", "high", "open",
                "close", "low", "zigzag_up",
                "zigzag_down"
            ]
        )
        self.data["date"] = self.data["date"] + " " + self.data["time"]
        self.data = self.data[["date", "zigzag_up", "zigzag_down"]]
        self.data.reset_index(drop=True, inplace=True)
        self.data["date"] = pd.to_datetime(self.data["date"], format="%Y.%m.%d")
        self.data = self.data.loc[
            (self.data["date"] >= "2017-01-01 00:00:00") &
            (self.data["date"] <= "2022-12-13 19:10:00")
        ]

    def format_data(self):
        self.data["zigzag_down"] = self.data["zigzag_down"] * -1
        self.data["trend"] = self.data["zigzag_down"] + self.data["zigzag_up"]
        self.data = self.data.loc[self.data["trend"] != 0]
        self.data = self.data[["date", "trend"]]
        self.data.loc[self.data["trend"] < 0, "trend"] = 0
        self.data.loc[self.data["trend"] > 0, "trend"] = 1

    def generate_result_data(self):
        data_rows = []
        prev_row = {}
        for (index, row) in enumerate(zip(self.data["date"], self.data["trend"])):
            if index == 0:
                prev_row["date"] = row[0]
                prev_row["diff"] = row[1]
            else:
                data_rows.append({
                    "date_open": prev_row["date"],
                    "date_close": row[0],
                    "diff":  1 if prev_row["diff"] == 0 else 0
                })
                prev_row["date"] = row[0]
                prev_row["diff"] = row[1]
        return pd.DataFrame(data=data_rows)

    def run(self):
        self.read_data()
        self.format_data()
        result = self.generate_result_data()
        result.to_csv("zigzag_labeler.csv")

ProccessMetatraderData().run()