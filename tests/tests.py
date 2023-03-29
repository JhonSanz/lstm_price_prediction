import unittest
from label_data import Labeler
from datetime import datetime

class TestDataLabeler(unittest.TestCase):
    def setUp(self):
        self.labeler = Labeler("tests/nas100_test.csv", "tests/orders_test.csv")

    def get_operable_by_dates(self, data, date_open, date_close):
        return data.loc[
            (data["date"] >= date_open) &
            (data["date"] <= date_close)
        ]["operable"].values

    def test_label_data(self):
        data = self.labeler.run()
        self.assertTrue(all(
            self.get_operable_by_dates(data, "2017-07-04 14:50:00", "2017-07-04 15:30:00") ==
            ([1] * 5)  + ([0] * 0)
        ))
