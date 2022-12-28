import numpy as np
import numpy.linalg
import scipy.optimize
import itertools
import math

def SpherePriorCalculator(rx, ry, rz, h):
    mat_r = np.diag([rx**-2, ry**-2, rz**-2])
    h = h
    def SphereField(x):
        n = x.shape[0]
        prior = np.empty((n,1))
        for i in range(n):
            prior[i] = -h * (x[i].dot(mat_r).dot(x[i].T) - 1) / 2
        return prior
    def SphereDerivField(x):
        n = x.shape[0]
        prior = np.empty((n,3))
        for i in range(n):
            prior[i] = -h*x.dot(mat_r)
        return prior
    return SphereField, SphereDerivField

def TSPCov3D(x1,x2,hyp): # 3D Thin Plate Kernel function between d dimensional vectors x1 and x2
    # hyp[0] is R, max distance possible
    dist = np.linalg.norm(x1-x2)
    dist /= hyp[0]
    if dist>1:
        return 0
    k = 2*(dist**3) - 3*(dist**2) + 1
    return k # Returns normalized values! Between 0 and 1

def TSPCov_ithDerivative(i): #Base parameters
    def TSPCov3DDerivative(x1,x2,hyp): # 3D Thin Plate Kernel function between d dimensional vectors x1 and x2
        # hyp[0] is R, max distance possible
        dist = np.linalg.norm(x1-x2)
        dist /= hyp[0]
        s_dist = x1[i]-x2[i]
        return 6*s_dist*(dist-1)/(hyp[0]**2) # ALSO NORMALIZED BE CAREFUL
    return TSPCov3DDerivative

def Reflect(x,x0,n): #Reflect a point X into a point X' using plan of reference defined by X0 and n
    x0 = x0.reshape((1,-1))
    n = n.reshape((1,-1))
    t = (x-x0).dot(n.T)
    t /= np.linalg.norm(n)**2
    return x - 2 * t * n

def SumSymmetryTables(Table1, P1, P2, hyp, Table2 = None, model = TSPCov3D):
    if Table2 is None:
        Table2 = Table1
    sum = 0
    for k in range(0,np.size(Table1,axis=2)):
        sum += model(Table1[P1,:,k],Table2[P2,:,0],hyp)
        if k == 0:
            sum2 = sum
    return sum, sum2

def Kernel(X1, hyp, X2=None, model = TSPCov3D, sym_vector = None, vertical_rescaling = True, nosym_Out = False): #X1 and X2 are datasets of d dimensional vectors
    #sym vector is a list with all the symmetry planeson this format: n rows of n planes, [S, X0, n] where X0 and n describe the plane and S y a number between 0 and 1 
    Auto = False
    if X2 is None:
        Auto = True
    if sym_vector is not None: #If there is symmetry info
        planesofsymmetry = len(sym_vector[1]) # Amount of symmetry vectors
        comb_list = itertools.product([False,True],repeat=planesofsymmetry) #Create all possible combinations of plane of symmetry
        comb_list = [item for item in comb_list]
        X1refs = np.zeros((*X1.shape, 2**planesofsymmetry))
        if not Auto:
            X2refs = np.zeros((*X2.shape, 2**planesofsymmetry))
        for i,comb in enumerate(comb_list): #For every combination of planes
            x1r = X1 # Points that will be reflected
            if not Auto:
                x2r = X2
            for plane, involved in enumerate(comb): #I check wether the plane is involved
                if involved is True: # If the plane is involved in this combination...
                    x1r = Reflect(x1r,sym_vector[0],sym_vector[1][plane]) # ...I reflect point through that plane
                    if not Auto:
                        x2r = Reflect(x2r,sym_vector[0],sym_vector[1][plane])
            X1refs[:,:,i] = x1r
            if not Auto:
                X2refs[:,:,i] = x2r
        X1 = X1refs
        if not Auto:
            X2 = X2refs
    if Auto:
        X2 = X1

    rowsX1 = np.size(X1,0) # Amount of data in X1
    rowsX2 = np.size(X2,0) # Amount of data in X2
    K = np.zeros((rowsX1, rowsX2)) # Create resulting K matrix
    if nosym_Out:
        Knon = np.zeros((rowsX1, rowsX2))

    j_start = 0
    for i in range(0, rowsX1):
        if Auto:
            j_start = i
        if sym_vector is not None and vertical_rescaling:
            ai = math.sqrt(SumSymmetryTables(X1, i, i, hyp, model=model)[0])
        for j in range(j_start, rowsX2):
            if sym_vector is not None:
                k, knon = SumSymmetryTables(X1, i, j, hyp, Table2=X2, model=model)
                if vertical_rescaling:
                    k /= ai
                    k /= math.sqrt(SumSymmetryTables(X2, j, j, hyp, model=model)[0])
            else:
                k = knon = model(X1[i,:],X2[j,:],hyp)
            K[i][j] = k
            if nosym_Out:
                Knon[i][j] = knon
            if(i!=j) and Auto: # K is a symmetric matrix
                K[j][i] = k
                if nosym_Out:
                    Knon[j][i] = knon
    if nosym_Out:
        return K, Knon
    else:
        return K

class GP:
    # GP Model
    Y = None
    X = None
    L = None
    hyp = None
    model = TSPCov3D
    sym_vector = None
    vertical_rescaling = True
    prior = None

    # Input Data
    Kx = None
    Kxx = None
    Xx = None

    # Measurements
    mu = None
    cov = None

    def ChangeParam(self, X, Y, L, hyp, model = TSPCov3D, sym_vector = None, vertical_rescaling = True):
        self.X = X
        self.Y = Y
        self.L = L
        self.hyp = hyp
        self.model = model
        self.sym_vector = sym_vector
        self.vertical_rescaling = vertical_rescaling
        self.ClearInputData()

    def ClearInputData(self):
        self.Kx = None
        self.Xx = None
        self.Kxx = None

        self.mu = None
        self.cov = None

    def NewInput(self, Xx, Kx = None, Kxx = None):
        self.ClearInputData()
        self.Xx = Xx
        self.Kx = Kx
        self.Kxx = Kxx

    def Mu(self, model = None): #Calculate GP resulting mean over Xx
        if self.L is None or self.L.size == 0:
            return np.array([0.0])
        if model == self.model or model is None:
            model = self.model
            Kx = self.Kx
            mu = self.mu
            different_model = False
        else:
            Kx = None # will need to reset Kx calculation if change of model (deriv)
            mu = None
            different_model = True

        if Kx is None:
            Kx = Kernel(self.Xx, self.hyp, X2=self.X, model=model,sym_vector=self.sym_vector, vertical_rescaling=self.vertical_rescaling)
        if mu is None:
            mu = Kx.dot(np.linalg.lstsq(self.L.T, np.linalg.lstsq(self.L, self.Y, rcond=None)[0],rcond=None)[0])
            if not different_model:
                self.Kx = Kx
                self.mu = mu
        return mu

    def Cov(self, simple=True, block = None): #Calculate GP resulting covar over Xx, if simple, I don't calculate or use Kxx just identity (since i assume 1 diagonal always)
        #Block allows to take a smaller block size n since cholezky works. Will rregress cov only over first n X
        if simple:
            Kxx = 1
        elif self.Kxx is None:
            self.Kxx = Kernel(self.Xx, self.hyp, model=self.model,sym_vector=self.sym_vector, vertical_rescaling = self.vertical_rescaling)
            Kxx = self.Kxx
        
        if self.L is None or self.L.size == 0:
            if simple:
                n = self.Xx.shape[0]
                return np.ones((n,n))
            return Kxx
        
        if block is not None:
            X = self.X[:block,:]
            L = self.L[:block,:block]
        else:
            L = self.L
        if self.Kx is None:
            self.Kx = Kernel(self.Xx, self.hyp, X2=self.X, model=self.model,sym_vector=self.sym_vector, vertical_rescaling=self.vertical_rescaling)
        Kx = self.Kx
        if block is not None:
            Kx = Kx[:,:block]

        if block is not None or self.cov is None:
            v = np.linalg.lstsq(L, Kx.T, rcond=None)[0]
            cov = Kxx - v.T.dot(v)
            if block is None:
                self.cov = cov
        else:
            return self.cov
        return cov

class GPOptHelper:
    input_dicts = {}

    def __init__(self, X, Y, L, hyp, model = TSPCov3D, sym_vector = None, vertical_rescaling = True, prior = None, block = None):
        self.input_dicts = {}
        self.X = X
        self.Y = Y
        self.L = L
        self.hyp = hyp
        self.model = model
        self.sym_vector = sym_vector
        self.vertical_rescaling = vertical_rescaling
        self.prior = prior
        self.block = block

    def GetGP(self,theta):
        theta = theta.ravel()
        input_key = theta.tobytes()
        if input_key in self.input_dicts:
            return self.input_dicts[input_key]
        else:
            gp = GP()
            gp.ChangeParam(self.X, self.Y, self.L, self.hyp, model = self.model, sym_vector = self.sym_vector, vertical_rescaling = self.vertical_rescaling)
            theta = CylinderToCartesian(theta).reshape((1,-1))
            gp.NewInput(theta)
            self.input_dicts[input_key] = gp # Will input a prior of 0 (initially)
            return gp

    def OptMu(self, theta):
        gp = self.GetGP(theta)
        mu = gp.Mu().ravel()[0]
        if self.prior is not None:
            mu += self.prior(gp.Xx).ravel()[0]
        return mu

    def OptNegCov(self, theta):
        gp = self.GetGP(theta)
        cov = -gp.Cov(block=self.block).ravel()
        return cov

def SphereToCartesian(coords):
    r = coords[2]
    s1 = math.sin(coords[0])
    c1 = math.cos(coords[0])
    s2 = math.sin(coords[1])
    c2 = math.cos(coords[1])

    return np.array([r*c1*s2, r*s1*s2, r*c2])

def CylinderToCartesian(coords):
     r = coords[0]
     c = math.cos(coords[1])
     s = math.sin(coords[1])

     return np.array([r*c, r*s, coords[2]])

def ML(Y, L): #Function that calculates ML. In order to optimize some hyp according to ML
    #An optimizable wrapper needs to be created like opt_GPcov, assembling the kernel K, with the hyperparameters that need to be optimized
    ML = 0
    ML -= 0.5 * Y.T.dot(np.linalg.lstsq(L.T, np.linalg.lstsq(L, Y, rcond=None)[0], rcond=None)[0])
    ML -=  0.5 * np.size(Y,0) * np.log(2*np.pi)
    ML -= np.sum(np.log(np.diag(L)))
    return ML
