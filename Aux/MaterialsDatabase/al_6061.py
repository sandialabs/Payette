''' Material properties file
    MATERIAL: 6061 T6 Aluminum
    SOURCE:   MatWeb (www.matweb.com)
    UNITS:    Kg/m/s/K
'''

parameters = {
    'RHO0':2700, # initial density
    'HARDB':95, # brinell hardness
    'HARDK':120, # knoop hardness
    'HARDRWA':40, # rockwell a hardness
    'HARDRWB':60, # rockwell b hardness
    'HARDV':107, # vickers hardness
    'Y':276e6, # tensile strength
    'YULT':310e6, # ultimate tensile strength
    'TAUY':207e6, # shear strength
    'YBULT':607e6, # ultimate bearing strength
    'YB':386e6, # bearing strength
    'PLM':114e6, # power law hardeing modulus
    'PLP':.42, # power law hardening power
    'SMOD':26.0e9, # shear modulus
    'DGDP':1.8, # derivative of shear modulus wrt pressure
    'DGDT':.000170016, # derivative of shear modulus wrt temperature
    'E':68.9e9, # young's modulus
    'NU':0.330, # poisson's ratio
    'CV':896, # specific heat
    'TM':855, # melt temperature
    'AL':167, # coef. of thermal expansion
    'GRPAR':1.97, # gruneisen parameter
    'S1':1.4, # linear slope us/up
    'C0':5240, # bulk wave speed
    }
