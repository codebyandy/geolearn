"""
:dataset: db.dataset is a container of data
"""

class Dataframe(object):
    def getGeo(self, ndigit=8):
        return self.lat, self.lon

    def getT(self):
        return self.time


class DataModel():
    def getDataTrain(self):
        return self.x, self.y, self.c
