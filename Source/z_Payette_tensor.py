import numpy
import scipy

from Payette_utils import *

nvec = 3
nsym = 6
ntens = 9

tol = 1.e-16

di3 = [[0,1,2],[0,1,2]]

I3x3 = numpy.eye(3)
Z3x3 = numpy.zeros((3,3))

Z3 = np.array([0., 0., 0.])
I3 = np.array([1., 1., 1.])
Z6 = np.array([0., 0., 0.,
                   0., 0.,
                       0.])
I6 = np.array([1., 0., 0.,
                   1., 0.,
                       1.])
Z9 = np.array([0., 0., 0.,
               0., 0., 0.,
               0., 0., 0.])
I9 = np.array([1., 0., 0.,
               0., 1., 0.,
               0., 0., 1.])

tens_map = { 0:(0,0), 1:(0,1), 2:(0,2),
             3:(1,0), 4:(1,1), 5:(1,2),
             6:(2,0), 7:(2,1), 8:(2,2) }
symtens_map = { 0:(0,0), 1:(0,1), 2:(0,2),
                         3:(1,1), 4:(1,2),
                                  5:(2,2) }
vec_map = { 0:(0,), 1:(1,), 2:(2,) }

class Tensor(numpy.ndarray):

    iam = "Tensor"

    def __new__(self, init=0.):

        new_tensor = numpy.ndarray.__new__(Tensor, (3,3))

        if isinstance(init,(float,int)):
            init = [init]*9

        elif isinstance(init,numpy.ndarray):
            if len(init.shape) == 1:
                if init.shape[0]  != 9:
                    message = "Length of Tensor must be 9, got {0}".format(len(init))
                    raise reportError(self.iam,message)
                pass

            elif len(init.shape) == 2:
                if init.shape != (3,3):
                    message = "Tensor must be 3x3, got {0}".format(init.shape)
                    raise reportError(self.iam,message)
                for rown, row in enumerate(init): new_tensor[rown:] = row[:]
                return new_tensor

            else:
                message = "Tensor must be 3x3, got {0}".format(init.shape)
                raise reportError(self.iam,message)

        elif not isinstance(init,(list,tuple)):
            message = "Got non list initial value for Tensor: {0}".format(init)
            raise reportError(self.iam,message)

        elif len(init) != 9:
            message = "Length of Tensor must be 9, got {0}".format(len(init))
            raise reportError(self.iam,message)

        for idx, val in enumerate(init): new_tensor[tens_map[idx]] = val

        return new_tensor

    def XX(self): return self[tens_map[0]]
    def XY(self): return self[tens_map[1]]
    def XZ(self): return self[tens_map[2]]
    def YX(self): return self[tens_map[3]]
    def YY(self): return self[tens_map[4]]
    def YZ(self): return self[tens_map[5]]
    def ZX(self): return self[tens_map[6]]
    def ZY(self): return self[tens_map[7]]
    def ZZ(self): return self[tens_map[8]]

    def asMIGArray(self):
        return numpy.array( [ self.XX(), self.XY(), self.XZ(),
                              self.YX(), self.YY(), self.YZ(),
                              self.ZX(), self.ZY(), self.ZZ() ] )
    def asMatrix(self):
        return numpy.array( [ [ self.XX(), self.XY(), self.XZ() ],
                              [ self.YX(), self.YY(), self.YZ() ],
                              [ self.ZX(), self.ZY(), self.ZZ() ] ] )
    def asOldArray(self):
        return numpy.array( [ self.XX(), self.YY(), self.ZZ(),
                              self.XY(), self.YZ(), self.XZ(),
                              self.YX(), self.ZY(), self.ZX() ] )

class SymTensor(Tensor,numpy.ndarray):

    iam = "SymTensor"

    def __new__(self, init=0.):

        new_symtensor = numpy.ndarray.__new__(SymTensor, (3,3))

        if isinstance(init,(float,int)):
            init = [init]*6

        elif isinstance(init,numpy.ndarray):
            if len(init.shape) == 1:
                if init.shape[0]  != 6:
                    message = ("Length of SymTensor must be 6, got {0}"
                               .format(len(init)))
                    raise reportError(self.iam,message)
                pass

            elif len(init.shape) == 2:
                if init.shape != (3,3):
                    message = "SymTensor must be 3x3, got {0}".format(init.shape)
                    raise reportError(self.iam,message)
                if ( abs(init[(1,0)] - init[(0,1)]) > tol or
                     abs(init[(2,0)] - init[(0,2)]) > tol or
                     abs(init[(2,1)] - init[(1,2)]) > tol ):
                    message = ("got non symmetric array to SymTensor: {0}"
                               .format(init))
                    raise reportError(self.iam,message)
                for rown, row in enumerate(init): new_symtensor[rown:] = row[:]
                return new_symtensor

            else:
                message = "Tensor must be 3x3, got {0}".format(init.shape)
                raise reportError(self.iam,message)

        elif not isinstance(init,(list,tuple)):
            message = "Got non list initial value for SymTensor: {0}".format(init)
            raise reportError(self.iam,message)

        if len(init) != 6:
            message = "Length of SymTensor must be 6, got {0}".format(len(init))
            raise reportError(self.iam,message)

        for idx, val in enumerate(init): new_symtensor[symtens_map[idx]] = val
        new_symtensor[(1,0)] = new_symtensor[(0,1)]
        new_symtensor[(2,0)] = new_symtensor[(0,2)]
        new_symtensor[(2,1)] = new_symtensor[(1,2)]

        return new_symtensor

    def XX(self): return self[symtens_map[0]]
    def XY(self): return self[symtens_map[1]]
    def XZ(self): return self[symtens_map[2]]
    def YX(self): return self[symtens_map[1]]
    def YY(self): return self[symtens_map[3]]
    def YZ(self): return self[symtens_map[4]]
    def ZX(self): return self[symtens_map[2]]
    def ZY(self): return self[symtens_map[3]]
    def ZZ(self): return self[symtens_map[5]]

    def asMIGArray(self):
        return numpy.array( [ self.XX(), self.YY(), self.ZZ(),
                              self.XY(), self.YZ(), self.XZ() ] )
    def asMatrix(self):
        return numpy.array( [ [ self.XX(), self.XY(), self.XZ() ],
                              [ self.XY(), self.YY(), self.YZ() ],
                              [ self.XZ(), self.YZ(), self.ZZ() ] ] )

class Vector(numpy.ndarray):

    iam = "Vector"

    def __new__(self, init=0.):

        new_vector = numpy.ndarray.__new__(Vector, (3,))

        if isinstance(init,(float,int)):
            init = [init]*3

        elif not isinstance(init,(list,tuple,numpy.ndarray)):
            message = "Got non list initial value for Vector: {0}".format(init)
            raise reportError(self.iam,message)

        if len(init) != 3:
            message = "Length of Vector must be 3, got {0}".format(len(init))
            raise reportError(self.iam,message)

        for idx, val in enumerate(init): new_vector[vec_map[idx]] = val

        return new_vector

    def X(self): return self[vec_map[0]]
    def Y(self): return self[vec_map[1]]
    def Z(self): return self[vec_map[2]]

    def asMIGArray(self):
        return numpy.array( [ self.X(), self.Y(), self.Z() ] )

def dot(a,b):
    """ dot product of a and b """
    adotb = numpy.dot(a,b)
    return returnTyp(adotb,a,b)

def inv(a):
    inva = numpy.linalg.inv(a)
    return returnTyp(inva,a)

def zeros(n):
    return numpy.zeros(n)

def det(a):
    return numpy.linalg.det(a)

def powm(a,m):
    aa = numpy.matrix(aa)
    if isdiag(aa):
        apowm = Z3x3
        apowm[di3] = numpy.diag(aa)**m
    else:
        w,v = numpy.linalg.eigh(aa)
        apowm = numpy.dot(numpy.dot(v,numpy.diag(w**m)),v.T)
        pass
    return returnTyp(apowm,a)

def expm(a,strict=False):
    aa = numpy.matrix(a)
    if isdiag(aa):
        expa = Z3x3
        expa[di3] = numpy.exp(numpy.diag(aa))
    elif strict:
        expa = numpy.real(scipy.linalg.expm(aa))
    else:
        expa = I3x3 + aa + numpy.dot(aa,aa)/2.
        pass
    return returnTyp(expa,a)

def sqrtm(a,strict=False):
    aa = numpy.matrix(a)
    if numpy.isnan(aa).any() or numpy.isinf(aa).any():
        msg = ("Probably reaching the numerical limits for the "
               "magnitude of the deformation.")
        reportError(__file__,msg)
        pass
    if isdiag(aa):
        sqrta = Z3x3
        sqrta[di3] = numpy.sqrt(numpy.diag(aa))
    elif strict:
        sqrta = numpy.real(scipy.linalg.sqrtm(aa))
    else:
        sqrta = powm(aa,0.5)
        pass
    return returnTyp(sqrta,a)

def logm(a,strict=False):
    aa = numpy.matrix(a)
    if isdiag(aa):
        loga = Z3x3
        loga[di3] = numpy.log(numpy.diag(aa))
    elif strict:
        loga = numpy.real(scipy.linalg.logm(aa))
    else:
        amI = aa - I3x3
        loga = amI - numpy.dot(amI,amI)/2. + numpy.dot(amI,numpy.dot(amI,amI))/3.
        pass
    return returnTyp(loga,aa)

def isdiag(a):
    return all([abs(x) <= tol for x in [       a[0,1],a[0,2],
                                        a[1,0],       a[1,2],
                                        a[2,0],a[2,1]       ]])

def returnTyp(a,*args):
    if all([isinstance(x,SymTensor) for x in args]): return SymTensor(a)
    elif any([isinstance(x,Vector) for x in args]): return Vector(a)
    elif any([isinstance(x,Tensor) for x in args]): return Tensor(a)
    else: return numpy.array(a)

