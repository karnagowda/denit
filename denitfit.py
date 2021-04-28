#Karna Gowda 
#0.1 20210216 Ported MATLAB code to Python
#0.2 20210420 Removed computation of CIs for rates, which doesn't work for Nar and Nir strains because only one parameter is varied/fit.
#0.2 20210421 Updated fitYields to fit an intercept as well, and treat Nar/Nir, Nar, and Nir differently. Changed CI to 68% (comparable to 1 standard error)
#0.2.1 20210428 Added the computation of R2 to fityields. Added a function to compute the RMSE of a global fit.
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from lmfit import Minimizer, conf_interval
import statsmodels.api as sm
from sklearn.metrics import r2_score
import copy

class experiment:
    #This class contains the experimental data for a given condition
    def __init__(self, ID, phen, N0, Nend, A0, A, I0, I, t):
        self.ID = ID
        self.phen = phen
        self.N0 = N0
        self.Nend = Nend
        self.A0 = A0
        self.A = A
        self.I0 = I0
        self.I = I
        self.t = t
        
def fitYields(experiments):
    if experiments[0].phen == 'Nar/Nir':
        DelA = np.array([])
        DelI = np.array([])
        DelOD = np.array([])
        for i in range(0,len(experiments)):
            DelA = np.append(DelA,experiments[i].A0-experiments[i].A[:,-1])
            DelI = np.append(DelI,experiments[i].A0-experiments[i].A[:,-1] + experiments[i].I0-experiments[i].I[:,-1])
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
        x = np.append(DelA.reshape(-1, 1), DelI.reshape(-1, 1),axis=1)
        x = np.append(x,np.ones((len(DelOD),1)),axis=1)
    if experiments[0].phen == 'Nar':
        DelA = np.array([])
        DelOD = np.array([])
        for i in range(0,len(experiments)):
            DelA = np.append(DelA,experiments[i].A0-experiments[i].A[:,-1])
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
        x = np.append(DelA.reshape(-1, 1), np.ones((len(DelOD),1)),axis=1)
    if experiments[0].phen == 'Nir':
        DelI = np.array([])
        DelOD = np.array([])
        for i in range(0,len(experiments)):
            DelI = np.append(DelI,experiments[i].I0-experiments[i].I[:,-1])
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
        x = np.append(DelI.reshape(-1, 1), np.ones((len(DelOD),1)),axis=1)

    mod = sm.OLS(DelOD, x)
    res = mod.fit()
    ci = res.conf_int(0.32)   # 68% confidence interval. comparable to 1 standard error
    
    #compute R2
    DelOD_pred = res.predict()
    r2 = r2_score(DelOD,DelOD_pred)
    
    if experiments[0].phen == 'Nar/Nir':
        gamA = res.params[0]
        gamI = res.params[1]
    elif experiments[0].phen == 'Nar':
        gamA = res.params[0]
        gamI = 0
        ci = np.array((ci[0],np.zeros(2),ci[1]))
    elif experiments[0].phen == 'Nir':
        gamA = 0
        gamI = res.params[0]
        ci = np.array((np.zeros(2),ci[0],ci[1]))

    return gamA, gamI, ci, r2
        
def fitRates(params,experiments,n=1):
    fitter = Minimizer(residualGlobLMFit, params, fcn_args=(experiments,))
    result_brute = fitter.minimize(method='brute')
    best_result = copy.deepcopy(result_brute)
    for candidate in result_brute.candidates:
        trial = fitter.minimize(method='leastsq', params=candidate.params)
        if trial.chisqr < best_result.chisqr:
            best_result = trial
    #compute 99% CI
    # ci = conf_interval(fitter, best_result,sigmas=[0.99])
    # return best_result, ci
    return best_result

def plotDenitFit(p,experiment,n=1, tick_labels=True, axes_labels=True):
    lines = np.array([[0.000,0.447,0.741],[0.850,0.325,0.098]])
    plt.rcParams.update({"text.usetex": False, 'font.size': 16});
    
    y0 = np.append(experiment.N0, [np.nanmedian(experiment.A[:,0]), np.nanmedian(experiment.I[:,0])])
    th = np.linspace(experiment.t[0],experiment.t[-1],256)
#     y0 = np.append(experiment.N0, [experiment.A0, experiment.I0])
#     th = np.linspace(0,experiment.t[-1],256)
    yh = denitODE(y0,th,p,n)
    
    plt.plot(th,yh[:,-2],'-',color=(lines[0,0],lines[0,1],lines[0,2]),linewidth=4,alpha=0.5)
    plt.plot(th,yh[:,-1],'-',color=(lines[1,0],lines[1,1],lines[1,2]),linewidth=4,alpha=0.5)
    plt.plot(experiment.t,experiment.A.transpose(),'o',color=(lines[0,0],lines[0,1],lines[0,2]))
    plt.plot(experiment.t,experiment.I.transpose(),'o',color=(lines[1,0],lines[1,1],lines[1,2]))
    if tick_labels==True:
        plt.xticks([0,8,16,32,64])
        plt.yticks([0,0.5,1,1.5,2])
    else:
        plt.xticks([])
        plt.yticks([])
        
    if axes_labels==True:
        plt.xlabel('time (h)')
        plt.ylabel('NO_2^-, NO_3^- (mM)')
    
def convertPTableToMat(params,n=1):
    if n == 1:
        p_out = [[params['rA'].value, params['rI'].value, params['kA'].value, params['kI'].value, params['gamA'].value, params['gamI'].value]]
    else:
        p_out = np.zeros((n,6))
        for i in range(0,n):
            p_out[i,0] = params[i]['rA'].value
            p_out[i,1] = params[i]['rI'].value
            p_out[i,2] = params[i]['kA'].value
            p_out[i,3] = params[i]['kI'].value
            p_out[i,4] = params[i]['gamA'].value
            p_out[i,5] = params[i]['gamI'].value
    return p_out

def residualGlobLMFit(params,experiments,n=1):
    return residualGlob(convertPTableToMat(params,n),experiments,n)

def residualGlob(p,experiments,n=1):
    #Computes the residual for all conditions
    res_out = np.array([])
    for i in range(0,len(experiments)):
        res_out = np.append(res_out,residual(p,experiments[i],n))
    return res_out

def RMSE(p,experiments,n=1):
    rmse_out = np.sqrt(np.mean((residualGlob(p,experiments,n))**2))
    return rmse_out

def residual(p,experiment,n=1):
    #Compute the residual vector for the A and I variables using the replicate measurements taken in a given condition
    y0 = np.append(experiment.N0, [np.nanmedian(experiment.A[:,0]), np.nanmedian(experiment.I[:,0])])
#     y0 = np.append(experiment.N0, [experiment.A0, experiment.I0])
    yh = denitODE(y0,experiment.t,p,n)
    return np.ravel([experiment.A-yh[:,-2],experiment.I-yh[:,-1]])

def denitODE(y0,t,p,n=1):
    sol = odeint(F, y0, t, Dfun=J, args=(p,n), rtol=1e-6)
    return sol

def F(y,t,p,n):
    #ODE RHS
    #takes p as an ndarray where each row contains the parameters for a given strain
    Fout = np.zeros(n+2)
    for i in range(0,n):
        Fout[ i]  = f(p[i],[y[i],y[-2],y[-1]])
        Fout[-2] += g(p[i],[y[i],y[-2],y[-1]])
        Fout[-1] += h(p[i],[y[i],y[-2],y[-1]])
    return Fout

def J(y,t,p,n):
    #Jacobian of F
    #takes p as an ndarray where each row contains the parameters for a given strain
    Jout = np.zeros((n+2,n+2))
    for i in range(0,n):
        Jout[ i, i]  = dfdN(p[i],[y[i],y[-2],y[-1]])
        Jout[ i,-2]  = dfdA(p[i],[y[i],y[-2],y[-1]])
        Jout[ i,-1]  = dfdI(p[i],[y[i],y[-2],y[-1]])

        Jout[-2, i]  = dgdN(p[i],[y[i],y[-2],y[-1]])
        Jout[-2,-2] += dgdA(p[i],[y[i],y[-2],y[-1]])

        Jout[-1, i]  = dhdN(p[i],[y[i],y[-2],y[-1]])
        Jout[-1,-2] += dhdA(p[i],[y[i],y[-2],y[-1]])
        Jout[-1,-1] += dhdI(p[i],[y[i],y[-2],y[-1]])
    return Jout

def f(p,y):
    #takes p as a 1darray
    rA     = p[0]
    rI     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    return gamA*rA*y[0]*y[1]/(kA + y[1]) + gamI*rI*y[0]*y[2]/(kI + y[2])

def g(p,y):
    #takes p as a 1darray
    rA     = p[0]
    kA     = p[2]
    return -1*rA*y[0]*y[1]/(kA + y[1])

def h(p,y):
    #takes p as a 1darray
    rA     = p[0]
    rI     = p[1]
    kA     = p[2]
    kI     = p[3]
    return rA*y[0]*y[1]/(kA + y[1]) - rI*y[0]*y[2]/(kI + y[2])

def dfdN(p,y):
    #takes p as a 1darray
    rA     = p[0]
    rI     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    return gamA*rA*y[1]/(kA + y[1]) + gamI*rI*y[2]/(kI + y[2])

def dfdA(p,y):
    #takes p as a 1darray
    rA     = p[0]
    kA     = p[2]
    gamA   = p[4]
    return gamA*rA*kA*y[0]/(kA + y[1])**2

def dfdI(p,y):
    #takes p as a 1darray
    rI     = p[1]
    kI     = p[3]
    gamI   = p[5]
    return gamI*rI*kI*y[0]/(kI + y[2])**2

def dgdN(p,y):
    #takes p as a 1darray
    rA     = p[0]
    kA     = p[2]
    return -1*rA*y[1]/(kA + y[1])

def dgdA(p,y):
    #takes p as a 1darray
    rA     = p[0]
    kA     = p[2]
    return -1*rA*kA*y[0]/(kA + y[1])**2

def dhdN(p,y):
    #takes p as a 1darray
    rA     = p[0]
    rI     = p[1]
    kA     = p[2]
    kI     = p[3]
    return rA*y[1]/(kA + y[1]) - rI*y[2]/(kI + y[2])

def dhdA(p,y):
    #takes p as a 1darray
    rA     = p[0]
    kA     = p[2]
    return rA*kA*y[0]/(kA + y[1])**2

def dhdI(p,y):
    #takes p as a 1darray
    rI     = p[1]
    kI     = p[3]
    return -1*rI*kI*y[0]/(kI + y[2])**2