""" Material properties file for the elastic material """

__all__ = {
    "al_6061": ["aluminum",],
    "tib2": ["titanium_diboride",],
    "ti_6al_4v": ["titanium",],
    }

al_6061 = {"units": "SI", "E": 68.9e9, "NU": 0.330, "Y": 276.e6}
tib2 = {"units": "SI", "E": 565e9, "NU": 0.108, "Y": 1.8e9}
ti_6al_4v = {"units": "SI", "E": 113.8e9, "NU": 0.342, "Y": 880.e6}

