import numpy as np

"""
TODO:
1. medias de 100 en 100 hasta 1000
2. distancia entre en close y cada media
3. promedio de todas las medias 
4. promedio de las distancias
5. diferencia entre el high y 3.
6. diferencia entre el low y 3.
7. desviación estándar de 100 en 100 hasta 1000
8. promedio de las desviaciones estándar
9. incremento de cada media (resta del valor actual y el anterior) 
10. incremento de cada desviación (resta del valor actual y el anterior)
"""
class DataDescriptor:
    LOW_LIMIT = 5
    MOVING_AVERAGES = 6
    STANDARD_DEVIATIONS = 6

    def __init__(self, data):
        self.data = data
    
    def add_moving_averrages(self, data):
        for i in range(self.LOW_LIMIT, self.MOVING_AVERAGES + 1):
            periods = i * 100
            data[f"sma_{periods}"] = (
                data.ta.sma(
                    close=data["close"], length=periods
                )
            )
        return data

    def distance_from_close_to_moving_average(self, data):
        for i in range(self.LOW_LIMIT, self.MOVING_AVERAGES + 1):
            periods = i * 100
            data[f"close_mean_{periods}_diff"] = (
                data["close"] - data[f"sma_{periods}"]
            )
        return data

    def average_of_moving_averrages(self, data):
        data["average_of_moving_averages"] = 0
        for i in range(self.LOW_LIMIT, self.MOVING_AVERAGES + 1):
            periods = i * 100
            data["average_of_moving_averages"] += data[f"sma_{periods}"]
        data["average_of_moving_averages"] /= self.MOVING_AVERAGES
        return data

    def average_of_distances_from_close(self, data):
        data["average_of_distances_from_close"] = 0
        for i in range(self.LOW_LIMIT, self.MOVING_AVERAGES + 1):
            periods = i * 100
            data["average_of_distances_from_close"] += data[f"close_mean_{periods}_diff"]
        data["average_of_distances_from_close"] /= self.MOVING_AVERAGES
        return data

    def distance_from_high_to_average_of_moving_averages(self, data):
        data["distance_from_high_to_average_of_moving_averages"] = (
            data["high"] - data["average_of_moving_averages"]
        )
        return data

    def distance_from_low_to_average_of_moving_averages(self, data):
        data["distance_from_low_to_average_of_moving_averages"] = (
            data["low"] - data["average_of_moving_averages"]
        )
        return data

    def add_standard_deviations(self, data):
        for i in range(self.LOW_LIMIT, self.STANDARD_DEVIATIONS + 1):
            periods = i * 100
            data[f"stdev_{periods}"] = (
                data.ta.stdev(
                    close=data["close"], length=periods
                )
            )
        return data
    
    def average_of_standard_deviations(self, data):
        data["average_of_standard_deviations"] = 0
        for i in range(self.LOW_LIMIT, self.STANDARD_DEVIATIONS + 1):
            periods = i * 100
            data["average_of_standard_deviations"] += data[f"stdev_{periods}"]
        data["average_of_standard_deviations"] /= self.STANDARD_DEVIATIONS
        return data

    def did_moving_average_increase(self, data):
        for i in range(self.LOW_LIMIT, self.MOVING_AVERAGES + 1):
            periods = i * 100
            data[f"did_moving_average_{periods}_increase"] = data[f"sma_{periods}"].diff()
        return data

    def did_standard_deviation_increase(self, data):
        for i in range(self.LOW_LIMIT, self.STANDARD_DEVIATIONS + 1):
            periods = i * 100
            data[f"did_standard_deviation_{periods}_increase"] = data[f"stdev_{periods}"].diff()
        return data

    def run(self):
        data = self.data.copy()
        data = self.add_moving_averrages(data)
        data = self.distance_from_close_to_moving_average(data)
        data = self.average_of_moving_averrages(data)
        data = self.average_of_distances_from_close(data)
        data = self.distance_from_high_to_average_of_moving_averages(data)
        data = self.distance_from_low_to_average_of_moving_averages(data)
        data = self.add_standard_deviations(data)
        data = self.average_of_standard_deviations(data)
        data = self.did_moving_average_increase(data)
        data = self.did_standard_deviation_increase(data)
        data.dropna(inplace=True)
        data.reset_index(inplace=True, drop=True)
        return data
