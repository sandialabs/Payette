''' Material properties file
    MATERIAL: 6061 T6 Aluminum
    SOURCE:   MatWeb (www.matweb.com)
    UNITS:    Kg/m/s/K
'''

parameters = {'initial density':2700,
              'brinell hardness':95,
              'knoop hardness':120,
              'rockwell a hardness':40,
              'rockwell b hardness':60,
              'vickers hardness':107,
              'tensile strength':276e6,
              'ultimate tensile strength':310e6,
              'shear strength':207e6,
              'ultimate bearing strength':607e6,
              'bearing strength':386e6,
              'hard power law coef':114e6,
              'hard power law power':.42,
              'shear modulus':26.0e9,
              'deriv shear modulus wrt pressure':1.8,
              'deriv shear modulus wrt temperature':.000170016,
              'youngs modulus':68.9e9,
              'poissons ratio':0.330,
              'specific heat':896,
              'melt temperature':855,
              'coef thermal exp':167,
              'gruneissen parameter':1.97,
              'lin slope us up':1.4,
              'bulk wave speed':5240}
