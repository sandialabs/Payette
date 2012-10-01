#!/usr/bin/env python
# Copyright (2011) Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain
# rights in this software.

# The MIT License

# Copyright (c) Sandia Corporation

# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import sys
from math import log, exp, sqrt

EPS = 2.2e-16

def old_2_new(a1,a2,a3,a4):
    '''
    NAME
        old_2_new

    PURPOSE
        Convert the old limit parameters to the new parameters

    INPUT
        a1,a2,a3,a4: old limit surface parameters

    OUTPUT
        stren: high-pressure strength (just a parameter if YSLOPE.ne.0)
        peaki1: peak tensile value of I1
        fslope: initial slope d(ROOTJ2)/d(I1) evaluated at I1=-PEAKI1
        yslope: high pressure slope

    AUTHORS
        Rebecca Brannon:theory, algorithm, and code
        Tim Fuller, python implementation
    '''

    huge = 1.e99

    if a2 == 0.:
        # Ff = (a1 - a3) - a4*I1
        if a4 == 0.:
            peaki1 = huge
        else:
            peaki1 = (a1 - a3)/a4
            pass
    else:
        if a4 == 0.:
            # Ff = a1 - a3*exp(a2*I1)
            if a3 == 0.:
                peaki1 = huge
            else:
                peaki1 = log(a1/a3)/a2
        elif a3 == 0.:
            # Ff = a1 - a4*I1
            peaki1 = a1/a4
        else:
            # all coefficients are nonzero. Iterate to find peaki1
            peaki1 = log(a1/a3)/a2      # first guess
            knt = 0
            while knt < 50:
                ffval = a1 - a3*exp(a2*peaki1) - a4*peaki1
                peaki1 = peaki1 - ffval/(-a2*a3*exp(a2*peaki1) - a4)
                knt += 1
                if EPS * ffval/(a1 - a3) + 1. != 1.: continue
                break
            if knt == 50: sys.exit('o2n.py iterate failure')
            pass
        pass

    # yield surface parameters
    stren = a1 - a4*peaki1
    fslope = -(-a2*a3*exp(a2*peaki1) - a4)
    yslope = a4

    return(stren,peaki1,fslope,yslope)

def new_2_old(stren,peaki1,fslope,yslope):
    '''
    NAME
        new_2_old

    PURPOSE
        Convert the new Kayenta limit parameters to the old parameters

    INPUT
        stren: high-pressure strength (just a parameter if YSLOPE.ne.0)
        peaki1: peak tensile value of I1
        fslope: initial slope d(ROOTJ2)/d(I1) evaluated at I1=-PEAKI1
        yslope: high pressure slope

    OUTPUT
        a1,a2,a3,a4: old limit surface parameters


    AUTHORS
        Rebecca Brannon:theory, algorithm, and code
        Tim Fuller, python implementation
    '''

    tiny = sqrt(EPS)/10.

    if stren == 0.:
        a1,a2,a3,a4 = [0.]*4
        if fslope != 0.:sys.exit('fslope must be zero if stren=0')
        if yslope != 0.:sys.exit('yslope must be zero if stren=0')
        pass
    else:
        a1 = stren + peaki1*yslope
        a2 = (fslope - yslope)/stren
        a3 = stren*exp(-peaki1*(fslope - yslope)/stren)
        a4 = yslope
        pass

    dum = max(stren,peaki1)
    if a1 < tiny*dum: a1 = 0.
    if a3 < tiny*dum: a3 = 0.
    if yslope > fslope: sys.exit('yslope needs to be smaller than fslope')
    if fslope - yslope < tiny: a2 = 0.

    if fslope == 0. and yslope == 0. and peaki1 > 1.e75:
        # user wants von mises
        a1 = stren
        a2,a3,a4 = [0.]*3
        pass

    if a2 == 0.:
        a1 = a1 - a3
        a3 = 0.

    return(a1,a2,a3,a4)

def dif(x,X):
    if X == 0.: dnom = 1.
    else: dnom = abs(X)
    return abs(X-x)/dnom > 5.e-6

if __name__ == '__main__':

    import os
    usage = ('''
NAME
    {0}

USAGE
    {0} <conversion type> [options] <strength parameters>

CONVERSION TYPES
    o2n     Convert old (a1,a2,a3,a4) strength parameters to new
    n2o     Convert new (stren,peaki1,fslope,yslope) strength parameters to old

OPTIONS
    -h      Print this message and exit
    -c      Run additional checks

EXAMPLE
    {0} o2n a1 a2 a3 a4
    {0} n2o stren peaki1 fslope yslope
'''.format(os.path.basename(sys.argv[0])))

    argv = sys.argv[1:]

    if "-h" in argv: sys.exit(usage)

    if argv[0] != 'o2n' and argv[0] != 'n2o':
        print('ERROR: must specify o2n or n2o')
        sys.exit(usage)
        pass

    if  '-c' in argv:
        check = True
        argv.remove("-c")
    else: check = False

    if len(argv[1:]) != 4:
        print('ERROR: wrong number of inputs')
        sys.exit(usage)
        pass

    # get parameters
    c1,c2,c3,c4 = [float(x) for x in argv[1:]]

    if argv[0] == 'o2n':
        stren,peaki1,fslope,yslope = old_2_new(c1,c2,c3,c4)
        print('stren =%12.5e, peaki1 =%12.5e, fslope =%12.5e, yslope =%12.5e\n'
              %(stren,peaki1,fslope,yslope))

        if check:
            a1,a2,a3,a4 = new_2_old(stren,peaki1,fslope,yslope)
            if dif(a1,c1) or dif(a2,c2) or dif(a3,c3) or dif(a4,c4):
                print('ERROR: conversions failed')
                print('GIVEN')
                print('a1 =%12.5e, a2 =%12.5e, a3 =%12.5e, '
                      'a4 =%12.5e\n'%(c1,c2,c3,c4))
                print('COMPUTED')
                print('a1 =%12.5e, a2 =%12.5e, a3 =%12.5e, '
                      'a4 =%12.5e\n'%(a1,a2,a3,a4))
                pass
            pass
        pass

    elif argv[0] == 'n2o':
        a1,a2,a3,a4 = new_2_old(c1,c2,c3,c4)
        print('a1 = %12.5e, a2 = %12.5e, a3 = %12.5e, a4 = %12.5e\n'
              %(a1,a2,a3,a4))

        if check:
            stren,peaki1,fslope,yslope = old_2_new(a1,a2,a3,a4)
            if dif(stren,c1) or dif(peaki1,c2) or dif(fslope,c3) or dif(yslope,c4):
                print('ERROR: conversions failed\n')
                print('GIVEN')
                print('stren =%12.5e, peaki1 =%12.5e, fslope =%12.5e, '
                      'yslope =%12.5e\n'%(c1,c2,c3,c4))
                print('COMPUTED')
                print('stren =%12.5e, peaki1 =%12.5e, fslope =%12.5e, '
                      'yslope =%12.5e\n'%(stren,peaki1,fslope,yslope))
                pass
            pass
        pass

    sys.exit(0)
