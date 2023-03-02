import numpy as np
import numpy.linalg
import scipy.optimize
import itertools
import math

def SpherePriorCalculator(rx, ry, rz, h, x0=None):
    mat_r = np.diag([rx**-2, ry**-2, rz**-2])
    h = h
    def SphereField(x):
        n = x.shape[0]
        if x0 is not None:
            x = np.copy(x)
            x -= x0.reshape((1,-1))
        prior = np.empty((n,1))
        for i in range(n):
            prior[i] = -h * (x[i].dot(mat_r).dot(x[i].T) - 1) / 2
        return prior
    def SphereDerivField(x):
        n = x.shape[0]
        if x0 is not None:
            x -= x0.reshape((1,-1))
        prior = np.empty((n,3))
        for i in range(n):
            x = np.copy(x)
            prior[i] = -h*x.dot(mat_r)
        return prior
    return SphereField, SphereDerivField

def CartesianDistance(x1, x2):
    return np.linalg.norm(x1-x2)

def CylindricalDistance(x1, x2):
    dist2 = x1[0]**2 + x2[0]**2 - 2* x1[0]* x2[0]* math.cos(x1[1]-x2[1])
    dist2 += (x1[2]-x2[2])**2
    return math.sqrt(dist2)

def TSPCov3D(dist, hyp): # 3D Thin Plate Kernel function for a distance
    dist /= hyp[0]
    if dist>1:
        return 0
    k = 2*(dist**3) - 3*(dist**2) + 1
    return k # Returns normalized values! Between 0 and 1

def Reflect(x,x0,n): #Reflect point(s) X into point(s) X' using plane of reference defined by X0 and n
    x0 = x0.reshape((1,-1))
    n = n.reshape((1,-1))
    t = (x-x0).dot(n.T)
    t /= np.linalg.norm(n)**2
    return x - 2 * t * n

def CombineTables(Table1, idx1, idx2, model, dist, hyp, mode = "sum", Table2 = None): # The second doesn't need the reflections (only values)
    if Table2 is None:
        Table2 = Table1
    value = 0
    for k in range(0,np.size(Table1,axis=2)): # Explore whole table 1
        current_value = model(dist(Table1[idx1,:,k],Table2[idx2,:,0]), hyp)
        if mode == "max":
            value = max(value, current_value)
        else:
            value += current_value
    return value

class GPX:
    Values = None
    SymTables = None
    RotTables = None
    RescaleTables = None
    def __init__(self, X = np.empty((0,3))):
        self.Values = X
    def __add__(self, other):
        result = GPX()
        result.Values = np.vstack((self.Values, other.Values))
        if self.SymTables is not None and other.SymTables is not None:
            result.SymTables = np.vstack((self.SymTables, other.SymTables))
        if self.RotTables is not None and other.RotTables is not None:
            result.RotTables = np.vstack((self.RotTables, other.RotTables))
        if self.RescaleTables is not None and other.RescaleTables is not None:
            result.RescaleTables = np.vstack((self.RescaleTables, other.RescaleTables))
        return result
    def length(self):
        return len(self.Values)

class GP:
    # region Y
    _Y = np.empty((0,1))
    @property
    def Y(self):
        return self._Y
    @Y.setter
    def Y(self, value):
        self._mu = None # Reset Y dependencies...
        self._ML = None
        self._Y = value
    #endregion
    # region hyp
    _hyp = None
    @property
    def hyp(self):
        return self._hyp
    @hyp.setter
    def hyp(self, value):
        self.ResetAll()
        self._hyp = value
    #endregion
    # region noise
    _noise = 0
    @property
    def noise(self):
        return self._noise
    @noise.setter
    def noise(self, value):
        self.ResetDataDependences(full = False) # Change everything in regression but not rescaling
        self._noise = value
    #endregion
    # region X
    _X = GPX() # X data
    def AddX(self, x):
        auxX = GPX(x)
        if self._K is not None: # Just need to update existing K...
            auxKxx = self.AutoKernel(auxX) + self.noise**2 * np.eye(auxX.length())
            auxKx = self.Kernel(self._X,auxX)
            self._K = np.block([[self._K, auxKx], [auxKx.T, auxKxx]]) # Add new elements regarding new X to kernel K
            self._L = None # And then reset all outputs...
            self._ML = None
            self._mu = None
            self._cov = None
        if self._X.SymTables is not None: # Update missing tables...
            auxX.SymTables = self.UpdateSymTables(auxX)
        if self._X.RotTables is not None:
            auxX.RotTables = self.UpdateRotTables(auxX)
        if self._X.RescaleTables is not None:
            auxX.RescaleTables = self.InitializeRescaleTables(auxX)
        if self._Kx is not None:
            auxKx = self.Kernel(self._Xx, auxX)
            self._Kx = np.hstack((self._Kx, auxKx))
        self._X += auxX
    def RemoveX(self, N): #n=How many LAST X I'll remove?
        nX = len(self._X.Values)-N # New number of elements in _X after this operation (will remove last N elements)
        self._X.Values = self._X.Values[:nX,:]
        if self._K is not None: # Need to remove last n rows and cols...
            self._K = self._K[:nX,:nX]
            if self._L is not None:
                self._L = self._L[:nX,:nX]
            self._ML = None
            self._mu = None
            self._cov = None
        if self._X.SymTables is not None: # Remove elements pertaining to last data
            self._X.SymTables = self._X.SymTables[:nX,:,:]
        if self._X.RotTables is not None:
            self._X.RotTables = self._X.RotTables[:nX,:,:]
        if self._X.RescaleTables is not None:
            self._X.RescaleTables = self._X.RescaleTables[:,nX]
        if self._Kx is not None:
            self._Kx = self._Kx[:,:nX]
        if self._Y is not None:
            self._Y = self._Y[:nX,:]
    #endregion
    # region sym_vector
    _SymVector = None # Reset Sym Vector
    @property
    def SymVector(self):
        return self._SymVector
    @SymVector.setter
    def SymVector(self, value):
        self.ResetAll()
        self._X.SymTables = self._X.RotTables = self._Xx.SymTables = self._Xx.RotTables = None
        self._SymVector = value
    #endregion
    # region rescaling
    _VerticalRescaling = False
    @property
    def VerticalRescaling(self):
        return self._VerticalRescaling
    @VerticalRescaling.setter
    def VerticalRescaling(self, value):
        self.ResetAll()
        self._VerticalRescaling = value
    #endregion
    # region rotation
    _RotationSymmetry = None
    @property
    def RotationSymmetry(self):
        return self._RotationSymmetry
    @RotationSymmetry.setter
    def RotationSymmetry(self, value):
        self.ResetAll()
        self._X.RotTables = self._Xx.RotTables = None
        self._RotationSymmetry = value
    #endregion
    # region simple sym model...
    _SimpleSymmetryModel = False
    @property
    def SimpleSymmetryModel(self):
        return self._SimpleSymmetryModel
    @SimpleSymmetryModel.setter
    def SimpleSymmetryModel(self, value):
        self.ResetAll()
        self._SimpleSymmetryModel = value
    #endregion
    # region Xx
    _Xx = GPX() # Xx data
    def AddXx(self, x):
        auxX = GPX(x)
        if self._Kxx is not None: # Just need to update existing K...
            if self._simpleKxx:
                self._Kxx = None # If simple just re-do it, doesn't cost anything
            else:
                auxKxx = self.AutoKernel(auxX)
                auxKx = self.Kernel(Xx,auxX)
                self._K = np.block([[self._Kxx, auxKx], [auxKx.T, auxKxx]]) # Add new elements regarding new X to kernel K
            self._mu = None # Then reset calculated outputs
            self._cov = None
        if self._Kx is not None: # Add elements to Kx...
            auxKx = self.Kernel(auxX, self._X)
            self._Kx = np.vstack((self._Kx, auxKx))
            self._mu = None # Then reset calculated outputs
            self._cov = None
        if self._Xx.SymTables is not None:
            auxX.SymTables = self.UpdateSymTables(auxX)
        if self._Xx.RotTables is not None:
            auxX.RotTables = self.UpdateRotTables(auxX)
        if self._Xx.RescaleTables is not None:
            auxX.RescaleTables = self.InitializeRescaleTables(auxX)

        self._Xx += auxX
    #endregion
    # region Kx
    _Kx = None
    @property
    def Kx(self):
        if self._Kx is None:
            if self._Xx.length()>0 and self._X.length()>0:
                if self._Xx.length() > self._X.length():
                    self._Kx = self.Kernel(self._X, self._Xx).T
                else:
                    self._Kx = self.Kernel(self._Xx, self._X)
        return self._Kx
    @Kx.setter
    def Kx(self, value):
        self.ResetInputDependences(full=False)
        self._Kx = value
    #endregion
    # region Kxx y simple_Kxx
    _simpleKxx = True
    @property
    def SimpleKxx(self):
        return self._simpleKxx
    @SimpleKxx.setter
    def SimpleKxx(self, value):
        self._Kxx = None # Only Reset Kxx
        self._simpleKxx = value
    
    _Kxx = None
    @property
    def Kxx(self):
        if self._Kxx is None:
            if self._Xx.length()>0:
                if self._simpleKxx:
                    n = self._Xx.length()
                    self._Kxx = np.ones((n,n))
                else:
                    self._Kxx = self.AutoKernel(self._Xx)
        return self._Kxx
    #endregion
    # region K
    _K = None
    @property
    def K(self):
        if self._K is None:
            if self._X.length()>0:
                self._K = self.AutoKernel(self._X) + self.noise**2 * np.eye(self._X.length())
        return self._K
    #endregion
    # region L
    _L = None
    @property
    def L(self):
        if self._L is None:
            if self.K is not None:
                self._L = np.linalg.cholesky(self.K)
        return self._L
    #endregion
    # region mu
    _mu = None
    @property
    def mu(self):
        if self._mu is None:
            self._mu = self.CalculateMu()
        return self._mu
    #endregion
    # region cov
    _cov = None
    @property
    def cov(self):
        if self._cov is None:
            self._cov = self.CalculateCov()
        return self._cov
    #endregion
    # region ML
    _ML = None
    @property
    def ML(self):
        if self._ML is None:
            self._ML = self.CalculateML()
        return self._ML
    #endregion

    def ResetAll(self):
        self.ResetInputDependences()
        self.ResetDataDependences()
    def ResetDataDependences(self, full = True):
        self._mu = None
        self._cov = None
        self._L = None
        self._ML = None
        self._K = None
        if full:
            self._X.RescaleTables = None
    def ResetInputDependences(self, full = True):
        self._mu = None
        self._cov = None
        self._Kx = None
        self._Kxx = None
        if full:
            self._Xx.RescaleTables = None

    def UpdateSymTables(self, x):
        if x.SymTables is None: # Is there a symmetric model?
            if self.SymVector is not None: 
                planesofsymmetry = len(self.SymVector[1]) # Amount of symmetry planes syms (where array is [point, [syms]])
                comb_list = itertools.product([False,True],repeat=planesofsymmetry) #Create all possible combinations of plane of symmetry
                comb_list = [item for item in comb_list]
                resulting_table = np.zeros((*x.Values.shape, 2**planesofsymmetry))
                for i, comb in enumerate(comb_list): #For every combination of planes e.g. 4 planes FF TF FT TT
                    x_reflected = x.Values # Start from the original point(s)
                    for plane, involved in enumerate(comb): #I check wether the plane is involved
                        if involved is True: # If the plane is involved in this combination...
                            x_reflected = Reflect(x_reflected,self.SymVector[0],self.SymVector[1][plane]) # ...I reflect point through that plane
                    resulting_table[:,:,i] = x_reflected
            else:
                resulting_table = np.zeros((*x.Values.shape, 1))
                resulting_table[:,:,0] = x.Values
            x.SymTables = resulting_table
        return x.SymTables

    def UpdateRotTables(self, x):
        if x.RotTables is None: # Is there rot table?
            if self.RotationSymmetry is not False: # Should there be?
                self.UpdateSymTables(x) # In this case i need also to check reflections
                resulting_table = np.zeros((x.SymTables.shape[0], x.SymTables.shape[1]-1, x.SymTables.shape[2])) # I start based on the reflected tables (even if no reflections, then will be same as x.Value)
                for i in range(x.SymTables.shape[2]): # i as which table to copy
                    for j in range(x.SymTables.shape[0]): # j as which point...
                        point = x.SymTables[j,:,i].ravel()
                        point = CartesianToRotSym(point, self.RotationSymmetry[0], self.RotationSymmetry[1])
                        resulting_table[j,:,i] = point.reshape(1,-1)
                x.RotTables = resulting_table
        return x.RotTables

    def InitializeRescaleTables(self, x):
        if x.RescaleTables is None:
            points = x.length()
            resulting_table = np.zeros(points)
            x.RescaleTables = resulting_table
        return x.RescaleTables

    def AutoKernel(self, X):
        sizeK = X.length() # Amount of points in X
        K = np.zeros((sizeK, sizeK)) # Create resulting K matrix

        # Choose where are the points, and distance function (cylinder coord vs cartesian)
        if self.RotationSymmetry:
            input_table = self.UpdateRotTables(X)
        else:
            input_table = self.UpdateSymTables(X)
        distance_function = CartesianDistance

        # If using simple symmetry (no summing of values, but instead max)
        rescaling_table = None
        mode = "sum"
        if self.SimpleSymmetryModel:
            mode = "max"
        elif self.VerticalRescaling:
            rescaling_table = self.InitializeRescaleTables(X)

        for level in range(sizeK): # Level between 0 and size-1
            for i in range(sizeK-level): # row between 0 and sizeK-level
                j = i + level
                k = CombineTables(input_table, i, j, TSPCov3D, distance_function, self.hyp, mode = mode)
                if rescaling_table is not None: # If i need to do rescaling, will read (and fill, if needed) the rescaling table
                    if rescaling_table[i] <= 0: # If yet empty, i know I am in the diagonal so it is easy to fill (since firt pass i=j always)
                        rescaling_table[i] = math.sqrt(k)
                    k /= rescaling_table[i]
                    k /= rescaling_table[j]
                K[i][j] = K[j][i] = k
        return K

    def Kernel(self, X1, X2):
        n1 = X1.length() # Amount of points in X1
        n2 = X2.length()
        K = np.zeros((n1, n2)) # Create resulting K matrix

        # Choose where are the points, and distance function (cylinder coord vs cartesian)
        if self.RotationSymmetry:
            input_table1 = self.UpdateRotTables(X1)
            input_table2 = self.UpdateRotTables(X2)
        else:
            input_table1 = self.UpdateSymTables(X1)
            input_table2 = self.UpdateSymTables(X2)
        distance_function = CartesianDistance
        
        # If using simple symmetry (no summing of values, but instead max)
        rescaling_table1 = None
        rescaling_table2 = None
        mode = "sum"
        if self.SimpleSymmetryModel:
            mode = "max"
        elif self.VerticalRescaling:
            rescaling_table1 = self.InitializeRescaleTables(X1)
            rescaling_table2 = self.InitializeRescaleTables(X2)

        for i in range(n1): # row between 0 and n1
            for j in range(n2):
                k = CombineTables(input_table1, i, j, TSPCov3D, distance_function, self.hyp, mode = mode, Table2=input_table2)
                if rescaling_table1 is not None and rescaling_table2 is not None: # If i need to do rescaling, will read (and fill, if needed) the rescaling table
                    if rescaling_table1[i] <= 0:
                        rescaling_table1[i] = math.sqrt(CombineTables(input_table1, i, i, TSPCov3D, distance_function, self.hyp, mode = mode))
                    if rescaling_table2[j] <= 0:
                        rescaling_table2[j] = math.sqrt(CombineTables(input_table2, j, j, TSPCov3D, distance_function, self.hyp, mode = mode))
                    k /= rescaling_table1[i]
                    k /= rescaling_table2[j]
                K[i][j] = k
        return K

    def CalculateMu(self): #Calculate GP resulting mean over Xx
        if self.L is None or self.L.size == 0:
            return np.zeros((self._Xx.length(), 1))
        mu = self.Kx.dot(np.linalg.lstsq(self.L.T, np.linalg.lstsq(self.L, self.Y, rcond=None)[0],rcond=None)[0])
        return mu

    def CalculateCov(self, simple=True): #Calculate GP resulting covar over Xx, if simple, I don't calculate or use Kxx just identity (since i assume 1 diagonal always)
        if self.L is None or self.L.size == 0:
            return self.Kxx
        
        v = np.linalg.lstsq(self.L, self.Kx.T, rcond=None)[0]
        cov = self.Kxx - v.T.dot(v)
        return cov

    def CalculateML(self): #Function that calculates ML. In order to optimize some hyp according to ML
        #An optimizable wrapper needs to be created like opt_GPcov, assembling the kernel K, with the hyperparameters that need to be optimized
        ML = 0
        ML -= 0.5 * self.Y.T.dot(np.linalg.lstsq(self.L.T, np.linalg.lstsq(self.L, self.Y, rcond=None)[0], rcond=None)[0])
        ML -=  0.5 * np.size(self.Y,0) * np.log(2*np.pi)
        ML -= np.sum(np.log(np.diag(self.L)))
        return ML

    def GetCopy(self):
        newGP = GP()
        newGP._hyp = self._hyp
        newGP._Y = self._Y
        newGP._noise = self._noise
        newGP._X = GPX(self._X.Values)
        newGP._SymVector = self._SymVector
        newGP._VerticalRescaling = self._VerticalRescaling
        newGP._RotationSymmetry = self._RotationSymmetry
        newGP._simpleKxx = self._simpleKxx
        newGP._SimpleSymmetryModel = self._SimpleSymmetryModel
        newGP._K = self._K
        newGP._L = self._L
        return newGP

def CylinderToCartesian(coords):
     r = coords[0]
     c = math.cos(coords[1])
     s = math.sin(coords[1])

     return np.array([r*c, r*s, coords[2]])

def CartesianToCylinder(coords):
    r = np.linalg.norm(coords[0:2])
    angle = np.arctan2(coords[1],coords[0])
    return np.array([r, angle, coords[2]])

def CartesianToRotSym(x,x0,v):
    x0 = x0.ravel()
    v = v.ravel()
    v_norm = np.linalg.norm(v)
    d = np.linalg.norm(np.cross(x-x0,v))/v_norm
    l = np.dot(x-x0,v)/v_norm
    return np.array([d,l])