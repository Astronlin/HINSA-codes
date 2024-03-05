import numpy as np
# import os,sys
import emcee,corner
import astropy.units as u
# from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots
plt.style.use(['science','nature'])

def tau_gaussian(x,vaxis):
    
    tau0, v0, sigma_v = x
    
    return tau0*np.exp(-(vaxis-v0)**2/2/sigma_v**2)

def rebin(array,rebin_n=2):

    if rebin_n<=1:
        return array
     
    array_trim  = array[0:int(len(array)/rebin_n)*rebin_n] # drop a tail of the array such that it can be grouped every n iterms

    array_rebin = np.nanmean(np.array(array_trim).reshape(-1,rebin_n),axis=1)

    return array_rebin
    
def HI_model_3comp(T_obs, tau, T_k=15, T_c=4.6, tau_f=0.1, p=0.9):
    
    T_b = (T_obs+(T_c-T_k)*(1-tau_f)*(1-np.exp(-tau)))/(1-p*(1-np.exp(-tau)))
    
    return T_b

def diff(y_values):

    return (y_values[1:]-y_values[0:-1])

########## Prior function ##########
def lnprior(x,x_priors):
    
    tau0, v0, sigma_v = x
   
    tau0_m, v0_0, sigma_v_0 = x_priors
    # Hard boundaries
    if tau0>0.0           and tau0<tau0_m        and \
       v0>v0_0-sigma_v_0  and v0<v0_0+sigma_v_0  and \
       sigma_v>0          and sigma_v<2*sigma_v_0:
        return - 0.5*(v0-v0_0)**2/0.1**2 - 0.5*(sigma_v-sigma_v_0)**2/0.1**2
    else:
        return -np.inf
    
########## Likelihood function ##########
def lnlike(x,T_obs,vaxis):

    tau   = tau_gaussian(x,vaxis)
    T_rec = HI_model_3comp(T_obs, tau)
    
    T_rec_dd     = diff(diff(T_rec))
    #T_rec_dd_std = np.nanstd(T_rec_dd)
    
    chi_sq = np.nansum(T_rec_dd**2)#/T_rec_dd_std**2)
    
    return -0.5*chi_sq

def ChiSqure(x,T_obs,vaxis):
    
    return -lnlike(x,T_obs,vaxis)

########## Posterior Probability function = Prior x Likelihood ##########
def lnprob(x,x_priors,T_obs,vaxis):
    lnp = lnprior(x,x_priors)
    if not np.isfinite(lnp):
        return -np.inf
    else:
        return lnlike(x,T_obs,vaxis) + lnp

def HINSA_fitting_MCMC(observed_spectrum,x_priors,out_prefix='myfitting',verbose=False):
    
    vaxis, T_obs = observed_spectrum
    
    ndim = 3
    
    # MCMC paramters
    nwalkers = 200
    nburnin  = 2000
    niter    = 2000
    # Walkers starting range

    tau0_m, v0_0, sigma_v_0 = x_priors

    tau0min     = 0.01
    tau0max     = tau0_m
    v0min       = v0_0-sigma_v_0
    v0max       = v0_0+sigma_v_0
    sigma_v_min = 0.5*sigma_v_0
    sigma_v_max = 1.5*sigma_v_0
    # Outputs
    outfile   = '{}_chain.txt'.format(out_prefix)
    outcorner = '{}_corner.pdf'.format(out_prefix)
    outmass   = '{}_MCMC.pdf'.format(out_prefix)

    #### RUNNING MCMC
    
    # Inizialize walkers in a random way
    p0 = []
    for i in range(nwalkers):
        # Set start positions for walkers
        pi = [np.random.uniform(tau0min     , tau0max)      , \
              np.random.uniform(v0min       , v0max)        , \
              np.random.uniform(sigma_v_min , sigma_v_max)]
        p0.append(pi)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=16, a=2.0,\
                                    args=[x_priors,T_obs,vaxis])

    # Burnin phase
    pos, prob, state = sampler.run_mcmc(p0, nburnin)
    sampler.reset()
    print("Burn-in completed")

    # MCMC run
    sampler.run_mcmc(pos, niter, rstate0=state)
    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
    print("MCMC complete. Now outputting data.")

    # Make convergence plot
    fig = plt.figure(figsize=(4, 4), dpi=300)
    gs = gridspec.GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    for i in range(0, nwalkers, 1):
        x = np.arange(0, niter ,1)
        y = sampler.lnprobability[i,:]
        ax0.plot(x, y, '.')
    plt.savefig('{}_Convergence.png'.format(out_prefix), bbox_inches='tight', format='png')
    plt.close()

    # # Write chain
    f = open(outfile, "w")
    chain = sampler.flatchain
    probab = sampler.flatlnprobability
    f.write("# tau0 v0 sigma_v lnLike\n")
    for i in range(0, nwalkers*niter):
        loglike = probab[i]
        # Exclude points points with log-likelihood=MNLNVAL
        if loglike > -1e200:
            f.write( '%f %f %f %f\n' % (chain[i,0], chain[i,1], chain[i,2], loglike) )
    f.close()

    #### ML VALUES

    # Exclude points with low LogLike
    # chain = np.delete(chain, np.where((loglike<-50.)), axis=0) # This line may be problematic. loglike -> probab ???

    # Calculate Quantiles
    tau0        = corner.quantile(chain[:,0], 0.5)
    tau0_low    = corner.quantile(chain[:,0], 0.16)
    tau0_upp    = corner.quantile(chain[:,0], 0.84)
    v0          = corner.quantile(chain[:,1], 0.5)
    v0_low      = corner.quantile(chain[:,1], 0.16)
    v0_upp      = corner.quantile(chain[:,1], 0.84)
    sigma_v     = corner.quantile(chain[:,2], 0.5)
    sigma_v_low = corner.quantile(chain[:,2], 0.16)
    sigma_v_upp = corner.quantile(chain[:,2], 0.84)

    etau0_upp    = tau0_upp-tau0
    etau0_low    = tau0_low-tau0
    ev0_upp      = v0_upp-v0
    ev0_low      = v0_low-v0
    esigma_v_upp = sigma_v_upp-sigma_v
    esigma_v_low = sigma_v_low-sigma_v

    if verbose:
        print("tau0: "        , tau0, etau0_low, etau0_upp)
        print("v0 (km/s): "   , v0, ev0_low,ev0_upp)
        print("sigma_v (km/s)", sigma_v,esigma_v_low,esigma_v_upp)

    #### MAKE CORNER PLOT
    
    MLvalue = [tau0,v0,sigma_v]
    #Sigma levels in 2D = 1-exp(-(x/s)**2/2)
    figure = corner.corner(chain, levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), quantiles=[0.16,0.5,0.84], labels=["$\\tau_{0}$", "$v_{0}$", "$\sigma_{\\rm v}$"], show_titles=True,label_kwargs={"fontsize": 15},title_kwargs={"fontsize": 15})
    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))
    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(MLvalue[i], color="r")
        ax.tick_params(labelsize=15)
    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(MLvalue[xi] , color="r")
            ax.axhline(MLvalue[yi] , color="r")
            ax.plot(MLvalue[xi]    , MLvalue[yi] , "sr")
            ax.tick_params(labelsize=15)

    figure.savefig(outcorner, bbox_inches='tight',format='pdf')
    plt.close()

    print('Completed MCMC :)')
    
    return tau0, v0, sigma_v

def HINSA_fitting_leastsq(observed_spectrum,x_priors,out_prefix='myfitting_lsq'):
    
    tau0_m, v0_0, sigma_v_0 = x_priors

    vaxis, T_obs = observed_spectrum 
    
    ini_p = np.array([0.4,v0_0,sigma_v_0])
    
    sol=minimize(ChiSqure,ini_p,args=(T_obs,vaxis),bounds=((0.01,1),(v0_0-sigma_v_0,v0_0+sigma_v_0),(sigma_v_0*0.9,sigma_v_0*1.1)))

    print(sol.x)
     
    return sol.x