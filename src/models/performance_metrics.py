import numpy as np

def calc_kge(yactual, ymodel):
    if ~(type(ymodel) == 'np.array'):
        ymodel = np.array(ymodel)
    if ~(type(yactual) == 'np.array'):
        yactual = np.array(yactual)
    
    r        = np.corrcoef(yactual, ymodel)[0,0]
    std_obs  = np.std(yactual)
    std_mod  = np.std(ymodel)
    mean_obs = np.mean(yactual)
    mean_mod = np.mean(ymodel)
    
    kge = 1 - np.sqrt((r - 1)**2 + (std_mod/std_obs - 1)**2 + (mean_mod/mean_obs - 1)**2 )
    
    return kge
    
# nash sutcliffe efficiency
def calc_nse(yactual, ymodel):
    if ~(type(ymodel) == 'np.array'):
        ymodel = np.array(ymodel)
    if ~(type(yactual) == 'np.array'):
        yactual = np.array(yactual)
        
    dif1    = (yactual - ymodel)
    dif1_sq = dif1**2
    
    dif2    = (yactual - np.mean(yactual))
    dif2_sq = dif2**2
    
    return 1 - np.sum(dif1_sq)/np.sum(dif2_sq)

def calc_nse_ln(yactual, ymodel):
    if ~(type(ymodel) == 'np.array'):
        ymodel = np.array(ymodel)
    if ~(type(yactual) == 'np.array'):
        yactual = np.array(yactual)
    
    yactual = np.log(yactual)
    ymodel  = np.log(ymodel)

    return calc_nse(yactual, ymodel)