import numpy as np
from numpy import random
from numpy import matlib
import scipy
from scipy.stats import nbinom

def gill_gamma_K(k,t_matrix,S,nCells,gamma_params):
    k = np.matlib.repmat(k,nCells,1)
    k[:,0] = np.random.gamma(gamma_params[0],1/gamma_params[1],size=nCells)

    num_t_pts = t_matrix.shape[1]
    X_mesh = np.empty((nCells,num_t_pts,2),dtype=int) #change to float if storing floats!!!!!!! 
    X_mesh[:] = np.nan

    t = np.zeros(nCells,dtype=float)
    tindex = np.zeros(nCells,dtype=int)

    #initialize state: integer unspliced, integer spliced 
    X = np.zeros((nCells,2))

    #initialize list of cells that are being simulated
    simindices = np.arange(nCells)
    activecells = np.ones(nCells,dtype=bool)

    while any(activecells):
        mu = np.zeros(nCells,dtype=int);
        n_active_cells = np.sum(activecells)
        
        (t_upd,mu_upd) = rxn_calculator_bdp2( \
            X[activecells,:], \
            t[activecells], \
            k[activecells,:], \
            n_active_cells)

        t[activecells] = t_upd
        mu[activecells] = mu_upd
        
        tvec_time = t_matrix[np.arange(n_active_cells),tindex[activecells]]
        update = np.zeros(nCells,dtype=bool)
        update[activecells] = t[activecells]>tvec_time
        
        while any(update):
            tobeupdated = np.where(update)[0]
            for i in range(len(tobeupdated)):
                X_mesh[simindices[tobeupdated[i]],tindex[tobeupdated[i]],:] = \
                    X[tobeupdated[i],:]
            
            tindex = tindex + update;
            ended_in_update = tindex[update]>=num_t_pts;

            if any(ended_in_update):
                ended = tobeupdated[ended_in_update];
                
                activecells[ended] = False;
                mu[ended] = 0;

                if ~any(activecells):
                    break            
            
            tvec_time = t_matrix[np.arange(n_active_cells),tindex[activecells]]
            update = np.zeros(nCells,dtype=bool)
            update[activecells] = t[activecells]>tvec_time
        
        z_ = np.where(activecells)[0]
        X[activecells] += S[mu[activecells]-1]
    return X_mesh

def gill_burst(k,t_matrix,S,nCells):
    k = np.matlib.repmat(k,nCells,1)

    num_t_pts = t_matrix.shape[1]
    X_mesh = np.empty((nCells,num_t_pts,2),dtype=int) #change to float if storing floats!!!!!!! 
    X_mesh[:] = np.nan

    t = np.zeros(nCells,dtype=float)
    tindex = np.zeros(nCells,dtype=int)

    #initialize state: integer unspliced, integer spliced 
    X = np.zeros((nCells,2))

    #initialize list of cells that are being simulated
    simindices = np.arange(nCells)
    activecells = np.ones(nCells,dtype=bool)

    while any(activecells):
        mu = np.zeros(nCells,dtype=int);
        n_active_cells = np.sum(activecells)
        
        (t_upd,mu_upd) = rxn_calculator_bdp2( \
            X[activecells,:], \
            t[activecells], \
            k[activecells,:], \
            n_active_cells)

        t[activecells] = t_upd
        mu[activecells] = mu_upd
        
        tvec_time = t_matrix[np.arange(n_active_cells),tindex[activecells]]
        update = np.zeros(nCells,dtype=bool)
        update[activecells] = t[activecells]>tvec_time
        
        while any(update):
            tobeupdated = np.where(update)[0]
            for i in range(len(tobeupdated)):
                X_mesh[simindices[tobeupdated[i]],tindex[tobeupdated[i]],:] = \
                    X[tobeupdated[i],:]
            
            tindex = tindex + update;
            ended_in_update = tindex[update]>=num_t_pts;

            if any(ended_in_update):
                ended = tobeupdated[ended_in_update];
                
                activecells[ended] = False;
                mu[ended] = 0;

                if ~any(activecells):
                    break            
            
            tvec_time = t_matrix[np.arange(n_active_cells),tindex[activecells]]
            update = np.zeros(nCells,dtype=bool)
            update[activecells] = t[activecells]>tvec_time
        
        z_ = np.where(activecells)[0]
        not_burst = mu[z_] > 1
        burst = mu[z_] == 1
        if any(not_burst):
            X[z_[not_burst]] += S[mu[z_[not_burst]]-1]
        if any(burst):                 
            bs = np.random.geometric(1/(1+S[0][0]),size=(sum(burst),1))-1
            X[z_[burst]] += np.matlib.hstack((bs,np.zeros((sum(burst),1),dtype=int)))
    return X_mesh

def rxn_calculator_bdp2(X,t,k,nCells):
    nRxn = 3

    kinit = k[:,0]
    beta = k[:,1]
    gamma = k[:,2]

    a = np.zeros((nCells,nRxn),dtype=float)
    a[:,0] = kinit
    a[:,1] = beta * X[:,0]
    a[:,2] = gamma * X[:,1]

    a0 = np.sum(a,1)
    t += np.log(1./np.random.rand(nCells)) / a0
    r2ao = a0 * np.random.rand(nCells)
    mu = np.sum(np.matlib.repmat(r2ao,nRxn+1,1).T >= np.cumsum(np.matlib.hstack((np.zeros((nCells,1)),a)),1) ,1)
    return (t,mu)

def cme_integrator(phys,M,N,t,marg):
    #phys = [kinit, bs, gamma, beta]
    #M = max nas
    #N = max mat
    if marg=='mature':
        M=1
    elif marg=='nascent':
        N=1
    NN = N
    N = 1+int(np.ceil((NN-1)/2))
    l=np.arange(M)
    k=np.arange(N)
    u_ = np.exp(-2j*np.pi*l/M)-1
    v_ = np.exp(-2j*np.pi*k/NN)-1
    u,v=np.meshgrid(u_,v_)
    u=u.flatten()
    v=v.flatten()
    
    fun = lambda x: INTFUNC_(x,phys[1:],u,v)

    INT = scipy.integrate.quad_vec(fun,0,t)
    I2 = INT[0]
    
    I = np.exp(I2*phys[0])
    I = np.reshape(I.T,(N,M))
    
    N_fin = N+1 if np.mod(NN-1,2)==0 else -1
    I=np.vstack((I,np.hstack((np.reshape(
        np.flipud(I[1:N_fin,0].conj()),(N_fin-2,1)),np.flip(I[1:N_fin,1:].conj(),(0,1))))))
    return np.real(np.fft.ifft2(I)).T
     
def INTFUNC_(x,params,U,V):
    f = params[2]/(params[2]-params[1])
    Ufun = f*V*np.exp(-params[1]*x) + (U-V*f)*np.exp(-params[2]*x)
    filt = ~np.isfinite(f)
    if np.any(filt): #params[1] = params[2]
    	Ufun[filt] = np.exp(-params[1]*x)*(U[filt] + params[1]*V[filt]*x)
    Ufun = params[0]*Ufun
    return Ufun/(1-Ufun)