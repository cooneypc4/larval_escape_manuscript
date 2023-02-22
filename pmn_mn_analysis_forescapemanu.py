#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:28:04 2022

@author: PatriciaCooney
"""
# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
from scipy import stats
import scipy.spatial as sp
from scipy.stats import levene

#%% make plotting functions to fix all the problems with updating same figure
#heatmap connectivity matrices
def plot_p2m_weights(mat_pm,muscles,allmuscs,names,ti):
    f,ax = plt.subplots()
    sb.heatmap(mat_pm)
    #option: sb.clustermap?
    ax.set(xlabel="Muscles (D-->V)", ylabel="PMNs", yticks=np.arange(len(names)), yticklabels=names, xticks = muscles, xticklabels = allmuscs, title = ti)
    #plt.xticks(fontsize=10, rotation=0)
    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)

    plt.tight_layout()
    plt.show()
    f.savefig(ti+'_p2m_dv_weights.svg', format = 'svg', dpi = 1200)

# #simplified sum of PMN inputs to D L or V muscles
# def plot_p2dlv_weights(mat_pd,grpnames,names):
#     f,ax = plt.subplots()
#     sb.heatmap(mat_pd)
#     #option: sb.clustermap?
#     ax.set(xlabel="Muscle Groups", ylabel="PMNs", yticks=np.arange(len(names)), yticklabels=names, xticklabels = grpnames, title = 'PMN Weights to General Muscle Groups')
    
#     plt.tight_layout()
#     plt.show()

# #circum plot
# def plot_musc_syndist(p2m,muscles,allmuscs,pnames):
#     f,ax = plt.subplots()
#     ax.barh(muscles, np.flip(allPMNs[p2m,:]))
#     ax.set(xlabel="Synaptic Weights", ylabel="Muscles (V --> D)", yticks = muscles, yticklabels = allmuscs[::-1], ylim = (1,30), title = pnames[p2m])
#     for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
#         tick.set_fontsize(8)
#     for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
#         tick.set_fontsize(8)
    
#     plt.tight_layout()
#     plt.show()
    
#PMN vs. wavg plot
def plot_xbar(wavgdist, shufsamp, regline, reglineshuf, ti):
    f,ax = plt.subplots()
    plt.plot(wavgdist, color = "blue", label="connectome")
    ax.set(xlabel="PMNs", ylabel="Weighted Average onto MNs",title = ti)
    for sa in np.arange(0,len(shufsamp)):
        plt.plot(shufsamp[:], color = "orange", alpha=0.3)
    plt.plot(regline, color = "black", label="connectome fit")
    plt.plot(reglineshuf, color = "red", label="shuffle fit")
    f.savefig(ti+'_xbar_dv_weights.svg', format = 'svg', dpi = 1200)
    
 
# #real vs. shuf variance measures boxplot
# def plot_var(sigmas,grpnames):
#     f,ax = plt.subplots()
#     plt.boxplot(sigmas, grpnames)
#     plt.xticks(ticks = [1,2], labels = grpnames)
#     ax.set(xlabel="Real vs. Shuffled Connectivity", ylabel="Variance of PMN outputs")
#     f.savefig(ti+'varboxplot_dv_weights.svg', format = 'svg', dpi = 1200)
    
#cosine similarity matrices
def plot_cos(cosmat,muscs,ti):
    f,ax = plt.subplots()
    sb.heatmap(cosmat, vmin=0, vmax=1)
    ax.set(xlabel="Muscles D-->V", ylabel="Muscles (V --> D)",xticks=np.arange(len(muscs)), xticklabels = muscs,  yticks=np.arange(len(muscs)), yticklabels = muscs,title = ti)
    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
        f.savefig(ti+'_cossim_weights.svg', format = 'svg', dpi = 1200)


#%% Load the data files with connectivity and NTs
Jpm = pd.read_excel('PMN to MN connectivity-matrix_correctroll.xlsx')
#Jpp = pd.read_excel('')
types = pd.read_excel('NT data 01 June.xlsx')

#%% Categorize the NT values
eNTs = ['Chat']
iNTs = ['Glut','GABA']

NTs = pd.Series(types.iloc[:,1])
NTvals = np.zeros(len(types))

einds = np.array(types.index[types['Chat'].isin(eNTs)])
iinds = np.array(types.index[types['Chat'].isin(iNTs)])

NTvals[einds] = 1
NTvals[iinds] = -1
#keep unknowns neutral as 0 and exclude from divided matrices
NTvals = NTvals[::2]

#%% Fix the naming issues
oldnames = np.array(types['A01c1_a1l'])
pnames = list()
ex_pnames = list()
in_pnames = list()

for ind,temp in enumerate(oldnames):
    if(ind%2) == 0:
        endname = temp.index('_')
        pnames.append(temp[:endname].lower().strip())
        if NTvals[int(ind/2)] == 1:
            ex_pnames.append(temp[:endname].lower().strip())
        elif NTvals[int(ind/2)] == -1:
            in_pnames.append(temp[:endname].lower().strip())


#%% weighted avg for each PMN according to DLV grps, sort, and replot
#fxn for generating and sorting PMNs according to weighted DLV sums

# 1. assign locations 1-30 - muscles
# 2. take weighted average for that spatial number for each PMN
# 3. sort PMN rows by where weighted average is highest for that PMN
def wavg(mat_syn, muscs):
    mat_out = np.zeros(len(mat_syn))
    var_out = np.zeros(len(mat_syn))
    
    for pm in np.arange(0,len(mat_syn)):
        pmint = int(pm)
        mat_out[pmint] = np.average(muscs, weights = mat_syn[pmint,:])
        var_out[pmint] = np.average((muscs - mat_out[pmint])**2, weights = mat_syn[pmint,:])
        
    sortpmns = mat_out.argsort()
    reordp = mat_syn[sortpmns,:]
    zreord = stats.zscore(reordp, axis = 1)
    
    xj = mat_out[sortpmns]
    
    #mat_out, sortpmns, reordp, zreord,
    
    return  xj, var_out, reordp, zreord

#%% fxn for the shuffle comparison
#shuffle weights matrix 1000x, choose PMN partners based on prob of MN input
def shufmat(cnxns,num_reps):
    rand_mats = [] #set this up to be a 1000d array; store each, then perform the wavg on each -- will extract mean xj or even plot all light then do avg dark; same with vars
    bicnxns = np.where(cnxns > 0, 1, 0)
    outputPMNs = np.sum(bicnxns,1) 
    totalMNin = np.sum(np.sum(bicnxns,0),0)
    inputMNs = (np.sum(bicnxns,0)) / totalMNin
    
    P = bicnxns.shape[0]
    M = bicnxns.shape[1]
    
    Wshuf = np.zeros([P,M])
    
    for rep in range(num_reps):
        for pout in range(P):
            outputs = np.random.choice(M, outputPMNs[pout], replace=False, p=inputMNs)
            Wshuf[pout,outputs] = 1
        rand_mats.append(Wshuf)
        Wshuf = np.zeros([P,M])
        
    return rand_mats
    
#%%
#draw regression lines and check if regressions are significantly different
import statsmodels.api as sm
def regressline(sample_xbar):
    X = sm.add_constant(np.arange(0,len(sample_xbar)))
    model = sm.OLS(sample_xbar,X)
    results = model.fit()
    params = model.fit().params
    
    #regline = params * np.arange(0,len(sample_xbar))
    regline = (params[0] + params[1] * X[:,1])
    
    return regline, results

# #define f test
# def f_test(group1, group2):
#     f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
#     nun = group1.size-1
#     dun = group2.size-1
#     p_value = 1-stats.f.cdf(f, nun, dun)
#     return f, p_value


#%% do all again but with the new MN order based on DLV and fxnal groups
#%% DLV grouping - this time 4,5,12 later than LTs; can also try with them before
dorfx = ['1','9','2','10','3','11','19','20']
latfx = ['18','24','23','22','21','8','25','26','27','29']
venfx = ['4','5','12','13','30','14','6','7','28','15','16','17']
allmuscsfx = dorfx + latfx + venfx

#%% reorder connectome according to this muscle order
#for each PMN (row) (look at that row and the row below, idx by 2 so that we're seeing L & R copies of PMNs)
epind = 0
ipind = 0

allPMNsfx = np.zeros([int(Jpm.shape[0]/2), len(allmuscsfx)])
ePMNsfx = np.zeros([len(ex_pnames), len(allmuscsfx)])
iPMNsfx = np.zeros([len(in_pnames), len(allmuscsfx)])

for pindie in np.arange(0,len(Jpm),2):
    #take sum of the two PMN-MN rows for each column
    lrpm = np.sum(np.array(Jpm.iloc[pindie:pindie+2,1:]),0).T
    for mi in np.arange(1,Jpm.shape[1],2):
        mind = list()
        #then take average of every two columns (L & R MNs) for each LR PMN pair
        pm = np.mean(lrpm[mi:mi+2])
        #for each MN name, check if number immediately preceded by 'MN' and suc by ' ' OR '/' (to check single vs. double digits)
        mtemp = Jpm.columns[mi+1]
        mtemp = mtemp.split('MN')[1]
        mtemp = mtemp[:mtemp.index(" ")]
        if '-' in mtemp:
            mtemp = mtemp[:mtemp.index("-")]
        #if '/' = 2 MNs, then store for the number strings before and after the '/'
        if '/' in mtemp:
            mtemp = mtemp.split('/')
            for im,mn in enumerate(mtemp):
                mind.append(allmuscsfx.index(mtemp[im])) #find idx of this mn in the allmuscs list
        else:
              mind = allmuscsfx.index(mtemp)   
        #store in PMN row MN col
        pind = int(pindie/2)
        allPMNsfx[pind,mind] = pm
        print('a_' + str(pindie) + '_' + str(pind) + ',' + str(mind))
        
        #break into E vs. I matrices
        if NTvals[int(pindie/2)] == 1:
            print('e_' + str(pindie) + '_' + str(epind) + ',' + str(mind))
            ePMNsfx[epind,mind] = pm
            if mind == 22:
                epind = epind + 1
            
        elif NTvals[int(pindie/2)] == -1:
            print('i_' + str(pindie) + '_' + str(ipind) + ',' + str(mind))
            iPMNsfx[ipind,mind] = pm
            if mind == 22:
                ipind = ipind + 1
                
#%% PMN DLV group plots
#plot grouped weights matrices
muscles = np.arange(0,30)
plot_p2m_weights(allPMNsfx,muscles,allmuscsfx,pnames,"All PMNs - reorganized muscles")
plot_p2m_weights(ePMNsfx,muscles,allmuscsfx,ex_pnames,"Excitatory PMNs - reorganized muscles")
plot_p2m_weights(iPMNsfx,muscles,allmuscsfx,in_pnames,"Inhibitory PMNs - reorganized muscles")

#%% weighted avg and sort and plot
xjallfx, sigjallfx, reordallfx, zreallfx, sortall = wavg(allPMNsfx, muscles)
xjexfx, sigjexfx, reordexfx, zreexfx, sortex = wavg(ePMNsfx, muscles)
xjinfx, sigjinfx, reordinfx, zreinfx, sortin = wavg(iPMNsfx, muscles)

plot_p2m_weights(reordallfx,muscles,allmuscsfx,pnames,"All PMNs - reorganized muscles")
plot_p2m_weights(zreallfx,muscles,allmuscsfx,pnames,"All PMNs - reorganized muscles")

plot_p2m_weights(reordexfx,muscles,allmuscsfx,ex_pnames,"Excitatory PMNs - reorganized muscles")
plot_p2m_weights(zreexfx,muscles,allmuscsfx,ex_pnames,"Excitatory PMNs - reorganized muscles")

plot_p2m_weights(reordinfx,muscles,allmuscsfx,in_pnames,"Inhibitory PMNs - reorganized muscles")
plot_p2m_weights(zreinfx,muscles,allmuscsfx,in_pnames,"Inhibitory PMNs - reorganized muscles")

#%% compare binary plots
biallfx = np.where(reordallfx>0, 1, 0)
biexfx = np.where(reordexfx>0, 1, 0)
biinfx = np.where(reordinfx>0, 1, 0)

plot_p2m_weights(biallfx,muscles,allmuscsfx,pnames,'PMN-MN Connectome Weights - All - reorg muscs')
plot_p2m_weights(biexfx,muscles,allmuscsfx,ex_pnames,'PMN-MN Connectome Weights - Excitatory - reorg muscs')
plot_p2m_weights(biinfx,muscles,allmuscsfx,in_pnames,'PMN-MN Shuffled Weights - Inhibitory - reorg muscs')


#%% shuffle matrices generation for all, e and i
randall = shufmat(allPMNsfx,1000)
randex = shufmat(ePMNsfx,1000)
randin = shufmat(iPMNsfx,1000)

#%%
#run the wavg fxn on the shuf mats
xjshufall = np.zeros([len(allPMNsfx),len(randall)])
xjshufex = np.zeros([len(ePMNsfx),len(randall)])
xjshufin = np.zeros([len(iPMNsfx),len(randall)])

varshufall = np.zeros([len(allPMNsfx),len(randall)])
varshufex = np.zeros([len(ePMNsfx),len(randall)])
varshufin = np.zeros([len(iPMNsfx),len(randall)])

matall = np.zeros([allPMNsfx.shape[0],allPMNsfx.shape[1],len(randall)])
matex = np.zeros([ePMNsfx.shape[0], ePMNsfx.shape[1],len(randall)])
matin = np.zeros([iPMNsfx.shape[0], iPMNsfx.shape[1],len(randall)])

zmatall = np.zeros([allPMNsfx.shape[0],allPMNsfx.shape[1],len(randall)])
zmatex = np.zeros([ePMNsfx.shape[0], ePMNsfx.shape[1],len(randall)])
zmatin = np.zeros([iPMNsfx.shape[0], iPMNsfx.shape[1],len(randall)])

for dim in np.arange(0,len(randall)):
    xjshufall[:,dim], varshufall[:,dim], matall[:,:,dim], zmatall[:,:,dim] = wavg(randall[dim], muscles)
    xjshufex[:,dim], varshufex[:,dim], matex[:,:,dim], zmatex[:,:,dim] = wavg(randex[dim], muscles)
    xjshufin[:,dim], varshufin[:,dim], matin[:,:,dim], zmatin[:,:,dim] = wavg(randin[dim], muscles)

#%% plot some of the shuffled matrices and see how they compare
pickrall = np.random.randint(0, matall.shape[2], size = 20)
pickrex = np.random.randint(0, matex.shape[2], size = 20)
pickrin = np.random.randint(0, matin.shape[2], size = 20)
#%%
# #compare xj's
mxj_all = np.mean(xjshufall, axis = 1)
mxj_ex = np.mean(xjshufex, axis = 1)
mxj_in = np.mean(xjshufin, axis = 1)

#%%
#draw regression lines and check if regressions are significantly different

def regressline(sample_xbar):
    if sample_xbar.ndim > 1:
        params = np.zeros([2,sample_xbar.shape[1]])
        regline = np.zeros([sample_xbar.shape[0],sample_xbar.shape[1]])
        for s in np.arange(0,sample_xbar.shape[1]):
            samp = sample_xbar[:,s]
            
            X = sm.add_constant(np.arange(0,len(samp)))
            model = sm.OLS(samp,X)
            params[:,s] = model.fit().params
            
            #regline = params * np.arange(0,len(sample_xbar))
            regline[:,s] = (params[0,s] + params[1,s] * X[:,1])
        
    else:    
        X = sm.add_constant(np.arange(0,len(sample_xbar)))
        model = sm.OLS(sample_xbar,X)
        params = model.fit().params

        #regline = params * np.arange(0,len(sample_xbar))
        regline = (params[0] + params[1] * X[:,1])
        
    
    return regline, params


#%%
# #regress and check if population variances of samples are equal, ftest for variances of populations
rall,mrall = regressline(xjallfx)
rshufall,mrshufall = regressline(mxj_all)
#varallvshuf = levene(mrall.resid, mrshufall.resid)
#fall = f_test(xjallfx,mxj_all)

rex,mrex = regressline(xjexfx)
rshufex,mrshufex = regressline(mxj_ex)
#varexvshuf = levene(mrex.resid, mrshufex.resid)
#fex = f_test(xjexfx,mxj_ex)

rin,mrin = regressline(xjinfx)
rshufin,mrshufin = regressline(mxj_in)
#varinvshuf = levene(mrin.resid, mrshufin.resid)
#fin = f_test(xjin,mxj_in)

#plot the data plus the regression lines
plot_xbar(xjallfx, xjshufall[:,pickrall], rall, rshufall, "All PMNs")
plot_xbar(xjexfx, xjshufex[:,pickrex], rex, rshufex, "Excitatory PMNs")
plot_xbar(xjinfx, xjshufin[:,pickrin], rin, rshufin, "Inhibitory PMNs")

#%% 
rshufall,paramshufall = regressline(xjshufall)
rshufin,paramshufin = regressline(xjshufin)
rshufex,paramshufex = regressline(xjshufex)

slopesall = mrall[1]
slopesex = mrex[1]
slopesin = mrin[1]

slopeshufall = paramshufall[1]
slopeshufex = paramshufex[1]
slopeshufin = paramshufin[1]

#%% use the histogram plotting for the regression slopes of xjs shuf vs real

def plotslopedist(alls,shufs,ti):
    dist = shufs
    #find 95% confidence
    ci = stats.t.interval(alpha=0.95, df=len(dist)-1, loc=np.mean(dist), scale=np.std(dist))
    perc95 = ci[1]
    perc5 = ci[0]
    
    #plot it
    f,ax = plt.subplots()
    plt.hist(dist, bins = 50, color = 'c', alpha = 0.6)
    plt.axvline(alls, color = 'm', linewidth = 2)
    plt.axvline(perc95, color = 'k', linestyle = 'dashed', linewidth = 1)
    plt.axvline(perc5, color = 'k', linestyle = 'dashed', linewidth = 1)
    ax.set(xlabel="Regression Slopes", ylabel="Frequency",title = ti)
    
    plt.tight_layout()
    plt.show()
    f.savefig(ti+'_regressslope_distribs.svg', format = 'svg', dpi = 1200)

#%%

plotslopedist(slopesall, slopeshufall, 'All PMNs')
plotslopedist(slopesex, slopeshufex, 'Excitatory PMNs')
plotslopedist(slopesin, slopeshufin, 'Inhibitory PMNs')

#%%
#pull out the connectivity for these
def findsubsets(inds,allmat,exmat,inmat):
    newmatall = list()
    newmatex = list()
    newmatin = list()
    for s in np.arange(0,len(allmat)):
        newmatall.append(allmat[s][:,inds])
        newmatex.append(exmat[s][:,inds])
        newmatin.append(inmat[s][:,inds])
    
    return newmatall, newmatex, newmatin
#%%
#pull out the connectivity for the shuffs
def findsubsetsshuf(inds):
    newshufall = np.zeros([len(matall),len(inds)])
    newshufex = np.zeros([len(matex),len(inds)])
    newshufin = np.zeros([len(matin),len(inds)])
    
    for i in np.arange(0,matall.shape[2]):
        newshufall = np.dstack((newshufall,matall[:,inds,i]))
        newshufex = np.dstack((newshufex,matex[:,inds,i]))
        newshufin = np.dstack((newshufin,matin[:,inds,i]))
    
    return newshufall, newshufex, newshufin

#%%
#run cossim and store ACS
def cossim(compmats):
    acsmat = np.zeros([len(compmats),len(compmats)])
    for m in np.arange(0,len(compmats)):
        j = m + 1
        selfcos = np.nanmean(1 - sp.distance.cdist(compmats[m].T, compmats[m].T, 'cosine'))
        selfmat = 1 - sp.distance.cdist(compmats[m].T, compmats[m].T, 'cosine')
        acsmat[m,m] = selfcos
        while j < len(compmats):
            othercos = np.nanmean(1 - sp.distance.cdist(compmats[m].T, compmats[j].T, 'cosine'))
            othermat = 1 - sp.distance.cdist(compmats[m].T, compmats[j].T, 'cosine')
            acsmat[m,j] = othercos
            acsmat[j,m] = othercos
            j = j + 1
    return acsmat


#%% plot distribution of ACS shuffles and highlight the real data
#pairwise cosine sim distrib histograms w/ real in red, shuf in gray
#draw dashed line at 95% of distribution - how to find % distrib of the real val?

# 15 plots, use for loop from the ACS mat fxn
#loop over the indices in the 5x5 ACS mat, loop over 1000 shuf its
def plot_cossim_dists(cosreal, cosshuf, title, groupnames):
    for i in np.arange(0,len(cosreal)):
        j = i
        while j < len(cosreal):
            tiplot = title + ': ' + groupnames[i] + ' x ' + groupnames[j]
            #get distrib of ACS
            dist = cosshuf[i,j,:]
            
            #find 95% confidence
            ci = stats.t.interval(alpha=0.95, df=len(dist)-1, loc=np.mean(dist), scale=np.std(dist))
            perc95 = ci[1]
            perc5 = ci[0]
     
            #plot it
            f,ax = plt.subplots()
            plt.hist(dist, bins = 50, color = 'c', alpha = 0.6)
            plt.axvline(cosreal[i,j], color = 'm', linewidth = 2)
            plt.axvline(perc95, color = 'k', linestyle = 'dashed', linewidth = 1)
            plt.axvline(perc5, color = 'k', linestyle = 'dashed', linewidth = 1)
            ax.set(xlabel="Cosine Similarity", ylabel="Frequency",title = tiplot)

            plt.tight_layout()
            plt.show()
            
            #next iter for this grp
            j = j + 1
#%% try cosine similarity comparisons of matrices -- shows sim of seq of numbers
#inner dot product of vectors divided by norm of vectors
#here, operate per PMN (colvecs) across MN grps (rows)
#so output is dorMNs to latMNs - 0 = orthog, 1 = overlap 100%

import scipy.spatial as sp

cosall = 1 - sp.distance.cdist(biallfx.T, biallfx.T, 'cosine')
cosex = 1 - sp.distance.cdist(biexfx.T, biexfx.T, 'cosine')
cosin = 1 - sp.distance.cdist(biinfx.T, biinfx.T, 'cosine')

#replot with muscle labels and appropriate titles
plot_cos(cosall, allmuscsfx, 'Cosine Similarity of Connections- All PMNs')
plot_cos(cosex, allmuscsfx, 'Cosine Similarity of Connections - Excitatory PMNs')
plot_cos(cosin, allmuscsfx, 'Cosine Similarity of Connections - Inhibitory PMNs')

d1 = ['1','9','2','10','3','11']
d1inds = np.arange(0,len(d1))
d2l1 = ['19','20','18','24','23','22','21','8']
d2inds = len(d1inds) + np.arange(0,len(d2l1))
v2 = ['25','26','27','29']
v2inds =  len(d1inds) + len(d2inds) + np.arange(0,len(v2))
l2v1 = ['4','5','12','13','30','14']
l2v1inds = len(d1inds) + len(d2inds) + len(v2inds) + np.arange(0,len(l2v1))
v3 = ['6','7','28','15','16','17']
v3inds = len(d1inds) + len(d2inds) + len(l2v1inds) + len(v2inds) + np.arange(0,len(v3))
newmuscorder = d1 + d2l1 + v2 + l2v1 + v3

d1matall, d1matex, d1matin = findsubsets(d1inds)
d2l1matall, d2l1matex, d2l1matin = findsubsets(d2inds)
v2matall, v2matex, v2matin = findsubsets(v2inds)
l2v1matall, l2v1matex, l2v1matin = findsubsets(l2v1inds)
v3matall, v3matex, v3matin = findsubsets(v3inds)

#%%
#calculate groupwise cosine similarities - real data and for loop shuff mats
acsall = cossim([d1matall,d2l1matall,v2matall,l2v1matall,v3matall])
acsex = cossim([d1matex,d2l1matex,v2matex,l2v1matex,v3matex])
acsin = cossim([d1matin,d2l1matin,v2matin,l2v1matin,v3matin])

#plot the cosine similarity matrices per muscle group
groupnames = ['D1','D2/L1','V1','L2/V2','V3']
plot_cos(acsall, groupnames,'Cosine Similarity Across Muscle Groups- All PMNs')
plot_cos(acsex, groupnames,'Cosine Similarity Across Muscle Groups - Excitatory PMNs')
plot_cos(acsin, groupnames,'Cosine Similarity Across Muscle Groups - Inhibitory PMNs')

#%%
#repeat for the pickr shuff subsets
#THEN PLOT THE DISTRIBUTION OF ACS VALS FOR THE SHUFF MATS, AND PLOT THE ACS VALS FOR REAL DATA IN THE DISTRIB IN DIFF COLOR

d1shufall, d1shufex, d1shufin = findsubsetsshuf(d1inds)
d2l1shufall, d2l1shufex, d2l1shufin = findsubsetsshuf(d2inds)
v2shufall, v2shufex, v2shufin = findsubsetsshuf(v2inds)
l2v1shufall, l2v1shufex, l2v1shufin = findsubsetsshuf(l2v1inds)
v3shufall, v3shufex, v3shufin = findsubsetsshuf(v3inds)

#%%
acsshufall = np.zeros([5,5,matall.shape[2]])
acsshufex = np.zeros([5,5,matall.shape[2]])
acsshufin = np.zeros([5,5,matall.shape[2]])
#get acs for all shufs
for s in np.arange(1,d1shufall.shape[2]):
    compshufall = [d1shufall[:,:,s],d2l1shufall[:,:,s],v2shufall[:,:,s],l2v1shufall[:,:,s],v3shufall[:,:,s]]
    compshufex = [d1shufex[:,:,s],d2l1shufex[:,:,s],v2shufex[:,:,s],l2v1shufex[:,:,s],v3shufex[:,:,s]]
    compshufin = [d1shufin[:,:,s],d2l1shufin[:,:,s],v2shufin[:,:,s],l2v1shufin[:,:,s],v3shufin[:,:,s]]
    acsshufall[:,:,s-1] = cossim(compshufall)
    acsshufex[:,:,s-1] = cossim(compshufex)
    acsshufin[:,:,s-1] = cossim(compshufin)

#plot the cosine similarity matrices per muscle group of example shuffle
groupnames = ['D1','D2/L1','V1','L2/V2','V3']
plot_cos(acsshufall[:,:,24], groupnames,'Cosine Similarity Across Muscle Groups- Shuffle All')
plot_cos(acsshufex[:,:,24], groupnames,'Cosine Similarity Across Muscle Groups - Shuffle Excitatory')
plot_cos(acsshufin[:,:,24], groupnames,'Cosine Similarity Across Muscle Groups - Shuffle Inhibitory')


#%% plot histograms for the ACS shufs vs real
grpnames = ['D1', 'D2L1','V1','L2V2','V3']

plot_cossim_dists(acsall,acsshufall,'All PMNs',grpnames)
plot_cossim_dists(acsex,acsshufex,'Excitatory PMNs',grpnames)
plot_cossim_dists(acsin,acsshufin,'Inhibitory PMNs',grpnames)

#%% ADDED 11/29 -- getting to bottom of why LTs diff and if crossed oscillator

#is there more PMN drive to LTs? binary and weigth distrib check by musc subgrp
#add up binary count and divide per num muscs in grp
# d1 = ['1','9','2','10','3','11']
# d1inds = np.arange(0,len(d1))
# d2l1 = ['19','20','18','24','23','22','21','8']
# d2inds = len(d1inds) + np.arange(0,len(d2l1))
# v2 = ['25','26','27','29']
# v2inds =  len(d1inds) + len(d2inds) + np.arange(0,len(v2))
# l2v1 = ['4','5','12','13','30','14']
# l2v1inds = len(d1inds) + len(d2inds) + len(v2inds) + np.arange(0,len(l2v1))
# v3 = ['6','7','28','15','16','17']
# v3inds = len(d1inds) + len(d2inds) + len(l2v1inds) + len(v2inds) + np.arange(0,len(v3))
# newmuscorder = d1 + d2l1 + v2 + l2v1 + v3



d1p = np.mean(biallfx[:,d1inds])
d2l1p = np.mean(biallfx[:,d2inds])
v2p = np.mean(biallfx[:,v2inds])
l2v1p = np.mean(biallfx[:,l2v1inds])
v3p = np.mean(np.mean(biallfx[:,v3inds]))
allmeaninput = [d1p, d2l1p, v2p, l2v1p, v3p]

def plot_meaninputs(allmeaninput, tiplot, groupnames):
    #plot it
    f,ax = plt.subplots()
    plt.bar(groupnames, allmeaninput, color = 'c', alpha = 0.8)
    ax.set(xlabel="Muscle Subgroups", ylabel="Number of PMN Inputs",title = tiplot)

    plt.tight_layout()
    plt.show()

plot_meaninputs(allmeaninput, 'Input to DV Functional Muscle Groups', grpnames)

# we see that LTs and VAs actually get less input than the other muscle groups (figure out how to test this statistically)

#%%
#plot syn weight distributions of the diff subgrps - plot syn weights in diff colors, semi-transparent all 
#OR plot each pairwise in 2 colors
def plot_syn_dists(alldistvec, title, groupnames):
    for i in np.arange(0,len(alldistvec)):
        j = i
        while j < len(alldistvec):
            tiplot = title + ': ' + groupnames[i] + ' x ' + groupnames[j]
            #get distrib of ACS
            disti = alldistvec[i]
            distj = alldistvec[j]
            
            #plot it
            f,ax = plt.subplots()
            plt.hist(disti, bins = 30, color = 'c', alpha = 0.4)
            plt.hist(distj, bins = 30, color = 'm', alpha = 0.4)
            
            ax.set(xlabel="PMN-MN Synaptic Weights", ylabel="Frequency",title = tiplot)

            plt.tight_layout()
            plt.show()
            
            #next iter for this grp
            j = j + 1

#collapse  each grp's weights into 1d vector
d1vec = np.reshape(allPMNsfx[:,d1inds][np.where(allPMNsfx[:,d1inds]>0.002)],-1) 
d2l1vec = np.reshape(allPMNsfx[:,d2inds][np.where(allPMNsfx[:,d2inds]>0.002)],-1)
v2vec = np.reshape(allPMNsfx[:,v2inds][np.where(allPMNsfx[:,v2inds]>0.002)],-1)
l2v1vec = np.reshape(allPMNsfx[:,l2v1inds][np.where(allPMNsfx[:,l2v1inds]>0.002)],-1)
v3vec = np.reshape(allPMNsfx[:,v3inds][np.where(allPMNsfx[:,v3inds]>0.002)],-1)
alldistvec = [d1vec, d2l1vec, v2vec, l2v1vec, v3vec]

plot_syn_dists(alldistvec,'Comparative Synaptic Weight Distributions', grpnames)

#%%
#%%
#%% 11/30
#Look at PMN-MN connections on L and R sides with cosine similarity analysis
#test quantitatively whether midline muscles show higher degree of LR shared PMN input
#test also if crossed oscillator hypothesis true--esp with LTs distinct timing from DLs/VLs but shared input on 2 sides; alt DL VL

#do this twice - once with the strict DV order and groupings that match Aref's PMN weights mat
#once with my current DV + functional mixing order of muscles

#%% first - with DV+functional groups I made
#reorg the PMN cnxn matrix into 4: LPMN-LMN, LPMN-RMN, RPMN-LMN, RPMN-RMN

#for each PMN (row) (look at that row and the row below, idx by 2 so that we're seeing L & R copies of PMNs)
epindLL = 0
epindLR = 0
epindRL = 0
epindRR = 0
ipindLL = 0
ipindLR = 0
ipindRL = 0
ipindRR = 0

LLallfx = np.zeros([int(Jpm.shape[0]/2), len(allmuscsfx)])
LRallfx = np.zeros([int(Jpm.shape[0]/2), len(allmuscsfx)])
RLallfx = np.zeros([int(Jpm.shape[0]/2), len(allmuscsfx)])
RRallfx = np.zeros([int(Jpm.shape[0]/2), len(allmuscsfx)])

eLLfx = np.zeros([len(ex_pnames), len(allmuscsfx)])
eLRfx = np.zeros([len(ex_pnames), len(allmuscsfx)])
eRLfx = np.zeros([len(ex_pnames), len(allmuscsfx)])
eRRfx = np.zeros([len(ex_pnames), len(allmuscsfx)])

iLLfx = np.zeros([len(in_pnames), len(allmuscsfx)])
iLRfx = np.zeros([len(in_pnames), len(allmuscsfx)])
iRLfx = np.zeros([len(in_pnames), len(allmuscsfx)])
iRRfx = np.zeros([len(in_pnames), len(allmuscsfx)])

for pindie in np.arange(0,len(Jpm)):
    for mi in np.arange(1,Jpm.shape[1]):
        pm = Jpm.iloc[pindie,mi]*2
        mind = list()
        
        #for each MN name, check if number immediately preceded by 'MN' and suc by ' ' OR '/' (to check single vs. double digits)
        mtemp = Jpm.columns[mi]
        mtemp = mtemp.split('MN')[1]
        mtemp = mtemp[:mtemp.index(" ")]
        if '-' in mtemp:
            mtemp = mtemp[:mtemp.index("-")]
        #if '/' = 2 MNs, then store for the number strings before and after the '/'
        if '/' in mtemp:
            mtemp = mtemp.split('/')
            for im,mn in enumerate(mtemp):
                mind.append(allmuscsfx.index(mtemp[im])) #find idx of this mn in the allmuscs list
        else:
            mind = allmuscsfx.index(mtemp)   
        
        #store in correct LR PMN-MN mat
        if pindie%2 == 0: #left even, right odd   
            pind = int(pindie/2)
            if mi%2 == 1: #left odd, right even
                LLallfx[pind,mind] = pm
                if NTvals[int(pindie/2)] == 1:
                    eLLfx[epindLL,mind] = pm
                    if '30' in mtemp:
                        epindLL = epindLL + 1
                elif NTvals[int(pindie/2)] == -1:
                    iLLfx[ipindLL,mind] = pm
                    if '30' in mtemp:
                        ipindLL = ipindLL + 1     
            else:
                LRallfx[pind,mind] = pm
                if NTvals[int(pindie/2)] == 1:
                    eLRfx[epindLR,mind] = pm
                    if '30' in mtemp:
                        epindLR = epindLR + 1
                elif NTvals[int(pindie/2)] == -1:
                    iLRfx[ipindLR,mind] = pm
                    if '30' in mtemp:
                        ipindLR = ipindLR + 1     
        else:
            pind = int((pindie-1)/2)
            if mi%2 == 1:
                RLallfx[pind,mind] = pm
                if NTvals[int(pindie/2)] == 1:
                    eRLfx[epindRL,mind] = pm
                    if '30' in mtemp:
                        epindRL = epindRL + 1
                elif NTvals[int(pindie/2)] == -1:
                    iRLfx[ipindRL,mind] = pm
                    if '30' in mtemp:
                        ipindRL = ipindRL + 1     
            else:
                RRallfx[pind,mind] = pm
                if NTvals[int(pindie/2)] == 1:
                    eRRfx[epindRR,mind] = pm
                    if '30' in mtemp:
                        epindRR = epindRR + 1
                elif NTvals[int(pindie/2)] == -1:
                    iRRfx[ipindRR,mind] = pm
                    if '30' in mtemp:
                        ipindRR = ipindRR + 1

#%%
d1 = ['1','9','2','10','3','11']
d1inds = np.arange(0,len(d1))
d2l1 = ['19','20','18','24','23','22','21','8']
d2inds = len(d1inds) + np.arange(0,len(d2l1))
v2 = ['25','26','27','29']
v2inds =  len(d1inds) + len(d2inds) + np.arange(0,len(v2))
l2v1 = ['4','5','12','13','30','14']
l2v1inds = len(d1inds) + len(d2inds) + len(v2inds) + np.arange(0,len(l2v1))
v3 = ['6','7','28','15','16','17']
v3inds = len(d1inds) + len(d2inds) + len(l2v1inds) + len(v2inds) + np.arange(0,len(v3))
newmuscorder = d1 + d2l1 + v2 + l2v1 + v3

#%%reorder all mats according to previously established wavg order, 
#binarize, and break into subsamples
def reordbinandsub(matin, sortord):
    matout = np.ndarray([matin[0].shape[0],matin[0].shape[1],int(len(matin))])
    for a in np.arange(0,len(matin)):
        matout[:,:,a] = np.where(matin[a][sortord]>0, 1, 0)
    
    return matout[:,:,0], matout[:,:,1], matout[:,:,2], matout[:,:,3]

allcomboLR = [LLallfx, LRallfx, RLallfx, RRallfx]
excomboLR = [eLLfx, eLRfx, eRLfx, eRRfx]
incomboLR = [iLLfx, iLRfx, iRLfx, iRRfx]

LLallfxbi, LRallfxbi, RLallfxbi, RRallfxbi = reordbinandsub(allcomboLR,sortall)
LLexfxbi, LRexfxbi, RLexfxbi, RRexfxbi = reordbinandsub(excomboLR,sortex)
LLinfxbi, LRinfxbi, RLinfxbi, RRinfxbi = reordbinandsub(incomboLR,sortin)

#plot binary weights
# plot_p2m_weights(LLallfxbi,muscles,allmuscsfx,pnames,'PMN-MN Connectome Weights - All - L PMN to L MN')
# plot_p2m_weights(LRallfxbi,muscles,allmuscsfx,pnames,'PMN-MN Connectome Weights - All - L PMN to R MN')
# plot_p2m_weights(RLallfxbi,muscles,allmuscsfx,pnames,'PMN-MN Shuffled Weights - All - R PMN to L MN')
# plot_p2m_weights(RRallfxbi,muscles,allmuscsfx,pnames,'PMN-MN Shuffled Weights - All - R PMN to R MN')

# plot_p2m_weights(LLexfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Connectome Weights - Excitatory - L PMN to L MN')
# plot_p2m_weights(LRexfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Connectome Weights - Excitatory - L PMN to R MN')
# plot_p2m_weights(RLexfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Shuffled Weights - Excitatory - R PMN to L MN')
# plot_p2m_weights(RRexfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Shuffled Weights - Excitatory - R PMN to R MN')

# plot_p2m_weights(LLinfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Connectome Weights - Inhibitory - L PMN to L MN')
# plot_p2m_weights(LRinfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Connectome Weights - Inhibitory - L PMN to R MN')
# plot_p2m_weights(RLinfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Shuffled Weights - Inhibitory - R PMN to L MN')
# plot_p2m_weights(RRinfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Shuffled Weights - Inhibitory - R PMN to R MN')

d1matall, d1matex, d1matin = findsubsets(d1inds, allcomboLR, excomboLR, incomboLR)
d2l1matall, d2l1matex, d2l1matin = findsubsets(d2inds, allcomboLR, excomboLR, incomboLR)
v2matall, v2matex, v2matin = findsubsets(v2inds, allcomboLR, excomboLR, incomboLR)
l2v1matall, l2v1matex, l2v1matin = findsubsets(l2v1inds, allcomboLR, excomboLR, incomboLR)
v3matall, v3matex, v3matin = findsubsets(v3inds, allcomboLR, excomboLR, incomboLR)


#%% multiside cossim fxn
#run cossim and store ACS
def cossim_multside(compmats):
    acsmat = np.zeros([len(compmats),len(compmats),len(compmats[0])])#output 1 mat for each side comp
    #first compare same muscle grp across the LL, LR, RL, and RR sides-store as first element 4 diff mats
    #and then concatenate into a bigger mat at the end
    for n in np.arange(0,len(compmats)): #loop thru muscle subgrps
        for m in np.arange(0,len(compmats[n])): #loop thru 4 sides         
            selfcos = np.nanmean(1 - sp.distance.cdist(compmats[n][m].T, compmats[n][m].T, 'cosine'))
            #selfmat = 1 - sp.distance.cdist(compmats[m].T, compmats[m].T, 'cosine')
            acsmat[n,n,m] = selfcos
            
            j = n + 1 #compare with neighbors
            while j < len(compmats):
                othercos = np.nanmean(1 - sp.distance.cdist(compmats[n][m].T, compmats[j][m].T, 'cosine'))
                #othermat = 1 - sp.distance.cdist(compmats[m].T, compmats[j].T, 'cosine')
                acsmat[n,j,m] = othercos
                acsmat[j,n,m] = othercos
                j = j + 1
            
    toleftmat = np.concatenate((acsmat[:,:,0],acsmat[:,:,2]),axis = 0)
    torightmat = np.concatenate((acsmat[:,:,1],acsmat[:,:,3]),axis = 0)
    bigmat = np.concatenate((toleftmat,torightmat),axis=1)
    
    collmat = np.concatenate(((acsmat[:,:,0] + acsmat[:,:,2])/2,(acsmat[:,:,1] + acsmat[:,:,3])/2), axis = 1)
            
    return acsmat, bigmat, collmat

#%%calculate groupwise cosine similarities - real data and for loop shuff mats
acsall, comball, collall = cossim_multside([d1matall,d2l1matall,v2matall,l2v1matall,v3matall])
acsex, combex, collex = cossim_multside([d1matex,d2l1matex,v2matex,l2v1matex,v3matex])
acsin, combin, collin = cossim_multside([d1matin,d2l1matin,v2matin,l2v1matin,v3matin])

#%%plot the cosine similarity matrices per muscle group
#cosine similarity matrices
def plot_cos_mult(cosmat1,cosmat2,muscs,ti):
    f,ax = plt.subplots()
    ax1 = plt.subplot(1,2,1)
    ax1 = sb.heatmap(cosmat1, vmin=0, vmax=1)
    ax1.set(xlabel="Muscles D-->V", ylabel="Muscles (V --> D)",xticks=np.arange(len(muscs)), xticklabels = muscs,  yticks=np.arange(len(muscs)), yticklabels = muscs,title = 'Ipsilateral')
    for tick in ax1.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    for tick in ax1.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
        
    ax2 = plt.subplot(1,2,2)
    ax2 = sb.heatmap(cosmat2, vmin=0, vmax=1)
    ax2.set(xlabel="Muscles D-->V", ylabel="Muscles (V --> D)",xticks=np.arange(len(muscs)), xticklabels = muscs, yticks=np.arange(len(muscs)), yticklabels = muscs,title = 'Contralateral')
    for tick in ax2.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    for tick in ax2.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
        
    plt.suptitle(ti)
    plt.show()

#%%
groupnames = ['D1','D2/L1','V1','L2/V2','V3']
plot_cos_mult(collall[:,0:5], collall[:,5:10], groupnames,'Cosine Similarity Across Muscle Groups- All PMNs')
plot_cos_mult(collex[:,0:5], collex[:,5:10], groupnames,'Cosine Similarity Across Muscle Groups - Excitatory PMNs')
plot_cos_mult(collin[:,0:5], collin[:,5:10], groupnames,'Cosine Similarity Across Muscle Groups - Inhibitory PMNs')

#%%generate shufmats for comparisons and pull out subsets of muscle groups
#shuffle weights matrix 1000x, choose PMN partners based on prob of MN input
def shufmat(cnxns,num_reps):
    rand_mats = [] #set this up to be a 1000d array; store each, then perform the wavg on each -- will extract mean xj or even plot all light then do avg dark; same with vars
    bicnxns = np.where(cnxns > 0, 1, 0)
    outputPMNs = np.sum(bicnxns,1) 
    totalMNin = np.sum(np.sum(bicnxns,0),0)
    inputMNs = (np.sum(bicnxns,0)) / totalMNin
    
    P = bicnxns.shape[0]
    M = bicnxns.shape[1]
    
    Wshuf = np.zeros([P,M])
    
    for rep in range(num_reps):
        for pout in range(P):
            outputs = np.random.choice(M, outputPMNs[pout], replace=False, p=inputMNs)
            Wshuf[pout,outputs] = 1
        rand_mats.append(Wshuf)
        Wshuf = np.zeros([P,M])
        
    return rand_mats

test = shufmat(LLallfxbi,1000)

randallLR = [shufmat(LLallfxbi,1000), shufmat(LRallfxbi,1000), shufmat(RLallfxbi,1000), shufmat(RRallfxbi,1000)]
randexLR = [shufmat(LLexfxbi,1000), shufmat(LRexfxbi,1000), shufmat(RLexfxbi,1000), shufmat(RRexfxbi,1000)]
randinLR = [shufmat(LLinfxbi,1000), shufmat(LRinfxbi,1000), shufmat(RLinfxbi,1000), shufmat(RRinfxbi,1000)]

#%%pull out the connectivity for the shuffs
def shufsub_cossim(inds,matin):   
    acsmat = np.zeros([len(inds),len(inds),len(matin),len(matin[0])])
    acscombmat = np.zeros([len(inds),len(inds),2,len(matin[0])])
    
    for l in np.arange(0,len(matin[0])): #1000shuffles
        for n in np.arange(0,len(inds)): #musc subgrps     
            for m in np.arange(0,len(matin)): #4 sides
                selfcos = np.nanmean(1 - sp.distance.cdist(matin[m][l][:,inds[n]].T, matin[m][l][:,inds[n]].T, 'cosine'))
                acsmat[n,n,m,l] = selfcos
                
                j = n + 1
                while j < len(inds):
                    othercos = np.nanmean(1 - sp.distance.cdist(matin[m][l][:,inds[n]].T, matin[m][l][:,inds[j]].T, 'cosine'))
                    acsmat[n,j,m,l] = othercos
                    acsmat[j,n,m,l] = othercos
                    j = j + 1
        
        acscombmat[:,:,0,l] = (acsmat[:,:,0,l] + acsmat[:,:,3,l])/2
        acscombmat[:,:,1,l] = (acsmat[:,:,1,l] + acsmat[:,:,2,l])/2
    
                
    return acscombmat #make this a 5,5,2,1000 array - w/in,w/o x w/in,w/o x 1000


acsshufallcoll = shufsub_cossim([d1inds,d2inds,v2inds,l2v1inds,v3inds], randallLR)
acsshufexcoll = shufsub_cossim([d1inds,d2inds,v2inds,l2v1inds,v3inds], randexLR)
acsshufincoll = shufsub_cossim([d1inds,d2inds,v2inds,l2v1inds,v3inds], randinLR)

            

d1shufall, d1shufex, d1shufin = findsubsetsshuf(d1inds)
d2l1shufall, d2l1shufex, d2l1shufin = findsubsetsshuf(d2inds)
v2shufall, v2shufex, v2shufin = findsubsetsshuf(v2inds)
l2v1shufall, l2v1shufex, l2v1shufin = findsubsetsshuf(l2v1inds)
v3shufall, v3shufex, v3shufin = findsubsetsshuf(v3inds)

#%%plot the cosine similarity matrices per muscle group of example shuffle
groupnames = ['D1','D2/L1','V1','L2/V2','V3']
plot_cos_mult(acsshufallcoll[:,:,0,24],acsshufallcoll[:,:,1,24], groupnames,'Cosine Similarity Across Muscle Groups- Shuffle All')
plot_cos_mult(acsshufexcoll[:,:,0,24],acsshufexcoll[:,:,1,24], groupnames,'Cosine Similarity Across Muscle Groups - Shuffle Excitatory')
plot_cos_mult(acsshufincoll[:,:,0,24],acsshufincoll[:,:,1,24], groupnames,'Cosine Similarity Across Muscle Groups - Shuffle Inhibitory')

#%% plot histograms for the ACS shufs vs real
def plot_cossim_dists(cosreal, cosshuf, title, groupnames):
    cosr1 = cosreal[:,0:5]
    cosr2 = cosreal[:,5:10]
    cosreal2d = [cosr1, cosr2]
    for m in np.arange(0,len(cosreal2d)):
        if m == 0:
            cat = ' - Ipsilateral'
        else:
            cat = ' - Contralateral'
        for i in np.arange(0,len(cosreal2d[0])):
            j = i
            while j < len(cosreal2d[0]):
                tiplot = title + cat + ': ' + groupnames[i] + ' x ' + groupnames[j]
                #get distrib of ACS
                dist = cosshuf[i,j,m,:]
            
                #find 95% confidence
                ci = stats.t.interval(alpha=0.95, df=len(dist)-1, loc=np.mean(dist), scale=np.std(dist))
                perc95 = ci[1]
                perc5 = ci[0]
         
                #plot it
                f,ax = plt.subplots()
                plt.hist(dist, bins = 50, color = 'c', alpha = 0.6)
                plt.axvline(cosreal2d[m][i,j], color = 'm', linewidth = 2)
                plt.axvline(perc95, color = 'k', linestyle = 'dashed', linewidth = 1)
                plt.axvline(perc5, color = 'k', linestyle = 'dashed', linewidth = 1)
                ax.set(xlabel="Cosine Similarity", ylabel="Frequency",title = tiplot)
        
                plt.tight_layout()
                plt.show()
                
                #next iter for this grp
                j = j + 1

#%%
grpnames = ['DL', 'D2L1','V1','L2V2','V3']

plot_cossim_dists(collall,acsshufallcoll,'All PMNs',grpnames)
plot_cossim_dists(collex,acsshufexcoll,'Excitatory PMNs',grpnames)
plot_cossim_dists(collin,acsshufincoll,'Inhibitory PMNs',grpnames)

#%%
#for stats options
#1. plot the cosine sim distrib's for each submat? do t-test or something? could also rep as boxplot and t tests, but then display as astersik on ACS mat?
#2. could do the shuf comparisons again--generate shuf mats on LR sided stats, take submats from those, calc ACS, 
#plot ACS distrib w/ CI's and show whether real data is outside CI
#prediction for 1 - mgiht not be stat powerful enough, but let's see
#prediciton for 2 - could see that more lat musc grps on L and R sides show ACS that falls within shuf dist, while midline muscles show ACS that is outside of shuf dist;
#could see that crossed oscillator muscles show within dist

#option 2 seems better, still allows comparison between groups but is way higher stat powered-- start with 2 then try 1. 
#still final fig maybe 2 examples then the ACS mat with asterisks on groups that were stat sig


#CONCLUSION SO FAR: WITH THIS GROUPING, WE SEE THAT IPSI AND CONTRA IS NOT CLEAN DIVIDE OF MIDLINE VS. NO. 
#SURPRISINGLY, LOOKS THAT SOME LATERAL GROUPS CUODL HAVE CROSS TALK, BUT THIS IS WHEN LTS ARE STLL LUMPED IN WITH DOS...
#%% second, with strict DV groups Aref made


#%%
#%%
#%%
#%%
#%%plot to check general match with the figure Aref made
#%% DLV grouping - AREF'S
dl = ['1','9','2','10']
dlinds = np.arange(0,len(dl))
do = ['3','4','11','19','20','5']
doinds = len(dlinds) + np.arange(0,len(do))
vl = ['12','13','14','30','6','7']
vlinds =  len(dlinds) + len(doinds) + np.arange(0,len(vl))
lt = ['18','8','21','22','23','24']
ltinds = len(dlinds) + len(doinds) + len(vlinds) + np.arange(0,len(lt))
va = ['25','26','27','29']
vainds = len(dlinds) + len(doinds) + len(vlinds) + len(ltinds) + np.arange(0,len(va))
vo = ['28','15','16','17']
voinds = len(dlinds) + len(doinds) + len(vlinds) + len(ltinds) + len(vainds) + np.arange(0,len(vo))

allmuscsdv = dl + do + vl + lt + va + vo

#reorg the PMN cnxn matrix into 4: LPMN-LMN, LPMN-RMN, RPMN-LMN, RPMN-RMN
#for each PMN (row) (look at that row and the row below, idx by 2 so that we're seeing L & R copies of PMNs)
epindLL = 0
epindLR = 0
epindRL = 0
epindRR = 0
ipindLL = 0
ipindLR = 0
ipindRL = 0
ipindRR = 0

LLalldv = np.zeros([int(Jpm.shape[0]/2), len(allmuscsdv)])
LRalldv = np.zeros([int(Jpm.shape[0]/2), len(allmuscsdv)])
RLalldv = np.zeros([int(Jpm.shape[0]/2), len(allmuscsdv)])
RRalldv = np.zeros([int(Jpm.shape[0]/2), len(allmuscsdv)])

eLLdv = np.zeros([len(ex_pnames), len(allmuscsdv)])
eLRdv = np.zeros([len(ex_pnames), len(allmuscsdv)])
eRLdv = np.zeros([len(ex_pnames), len(allmuscsdv)])
eRRdv = np.zeros([len(ex_pnames), len(allmuscsdv)])

iLLdv = np.zeros([len(in_pnames), len(allmuscsdv)])
iLRdv = np.zeros([len(in_pnames), len(allmuscsdv)])
iRLdv = np.zeros([len(in_pnames), len(allmuscsdv)])
iRRdv = np.zeros([len(in_pnames), len(allmuscsdv)])

for pindie in np.arange(0,len(Jpm)):
    for mi in np.arange(1,Jpm.shape[1]):
        pm = Jpm.iloc[pindie,mi]*2
        mind = list()
        
        #for each MN name, check if number immediately preceded by 'MN' and suc by ' ' OR '/' (to check single vs. double digits)
        mtemp = Jpm.columns[mi]
        mtemp = mtemp.split('MN')[1]
        mtemp = mtemp[:mtemp.index(" ")]
        if '-' in mtemp:
            mtemp = mtemp[:mtemp.index("-")]
        #if '/' = 2 MNs, then store for the number strings before and after the '/'
        if '/' in mtemp:
            mtemp = mtemp.split('/')
            for im,mn in enumerate(mtemp):
                mind.append(allmuscsdv.index(mtemp[im])) #find idx of this mn in the allmuscs list
        else:
            mind = allmuscsdv.index(mtemp)   
        
        #store in correct LR PMN-MN mat
        if pindie%2 == 0: #left even, right odd   
            pind = int(pindie/2)
            if mi%2 == 1: #left odd, right even
                LLalldv[pind,mind] = pm
                if NTvals[int(pindie/2)] == 1:
                    eLLdv[epindLL,mind] = pm
                    if '30' in mtemp:
                        epindLL = epindLL + 1
                elif NTvals[int(pindie/2)] == -1:
                    iLLdv[ipindLL,mind] = pm
                    if '30' in mtemp:
                        ipindLL = ipindLL + 1     
            else:
                LRalldv[pind,mind] = pm
                if NTvals[int(pindie/2)] == 1:
                    eLRdv[epindLR,mind] = pm
                    if '30' in mtemp:
                        epindLR = epindLR + 1
                elif NTvals[int(pindie/2)] == -1:
                    iLRdv[ipindLR,mind] = pm
                    if '30' in mtemp:
                        ipindLR = ipindLR + 1     
        else:
            pind = int((pindie-1)/2)
            if mi%2 == 1:
                RLalldv[pind,mind] = pm
                if NTvals[int(pindie/2)] == 1:
                    eRLdv[epindRL,mind] = pm
                    if '30' in mtemp:
                        epindRL = epindRL + 1
                elif NTvals[int(pindie/2)] == -1:
                    iRLdv[ipindRL,mind] = pm
                    if '30' in mtemp:
                        ipindRL = ipindRL + 1     
            else:
                RRalldv[pind,mind] = pm
                if NTvals[int(pindie/2)] == 1:
                    eRRdv[epindRR,mind] = pm
                    if '30' in mtemp:
                        epindRR = epindRR + 1
                elif NTvals[int(pindie/2)] == -1:
                    iRRdv[ipindRR,mind] = pm
                    if '30' in mtemp:
                        ipindRR = ipindRR + 1


#%%reorder all mats according to previously established wavg order, 
#binarize, and break into subsamples
def reordbinandsub(matin, sortord):
    matout = np.ndarray([matin[0].shape[0],matin[0].shape[1],int(len(matin))])
    for a in np.arange(0,len(matin)):
        matout[:,:,a] = np.where(matin[a][sortord]>0, 1, 0)
    
    return matout[:,:,0], matout[:,:,1], matout[:,:,2], matout[:,:,3]

allcomboLR = [LLalldv, LRalldv, RLalldv, RRalldv]
excomboLR = [eLLdv, eLRdv, eRLdv, eRRdv]
incomboLR = [iLLdv, iLRdv, iRLdv, iRRdv]

LLalldvbi, LRalldvbi, RLalldvbi, RRalldvbi = reordbinandsub(allcomboLR,sortall)
LLexdvbi, LRexdvbi, RLexdvbi, RRexdvbi = reordbinandsub(excomboLR,sortex)
LLindvbi, LRindvbi, RLindvbi, RRindvbi = reordbinandsub(incomboLR,sortin)

#plot binary weights
# plot_p2m_weights(LLallfxbi,muscles,allmuscsfx,pnames,'PMN-MN Connectome Weights - All - L PMN to L MN')
# plot_p2m_weights(LRallfxbi,muscles,allmuscsfx,pnames,'PMN-MN Connectome Weights - All - L PMN to R MN')
# plot_p2m_weights(RLallfxbi,muscles,allmuscsfx,pnames,'PMN-MN Shuffled Weights - All - R PMN to L MN')
# plot_p2m_weights(RRallfxbi,muscles,allmuscsfx,pnames,'PMN-MN Shuffled Weights - All - R PMN to R MN')

# plot_p2m_weights(LLexfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Connectome Weights - Excitatory - L PMN to L MN')
# plot_p2m_weights(LRexfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Connectome Weights - Excitatory - L PMN to R MN')
# plot_p2m_weights(RLexfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Shuffled Weights - Excitatory - R PMN to L MN')
# plot_p2m_weights(RRexfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Shuffled Weights - Excitatory - R PMN to R MN')

# plot_p2m_weights(LLinfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Connectome Weights - Inhibitory - L PMN to L MN')
# plot_p2m_weights(LRinfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Connectome Weights - Inhibitory - L PMN to R MN')
# plot_p2m_weights(RLinfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Shuffled Weights - Inhibitory - R PMN to L MN')
# plot_p2m_weights(RRinfxbi,muscles,allmuscsfx,ex_pnames,'PMN-MN Shuffled Weights - Inhibitory - R PMN to R MN')

dlmatall, dlmatex, dlmatin = findsubsets(dlinds, allcomboLR, excomboLR, incomboLR)
domatall, domatex, domatin = findsubsets(doinds, allcomboLR, excomboLR, incomboLR)
vlmatall, vlmatex, vlmatin = findsubsets(vlinds, allcomboLR, excomboLR, incomboLR)
ltmatall, ltmatex, ltmatin = findsubsets(ltinds, allcomboLR, excomboLR, incomboLR)
vamatall, vamatex, vamatin = findsubsets(vainds, allcomboLR, excomboLR, incomboLR)
vomatall, vomatex, vomatin = findsubsets(voinds, allcomboLR, excomboLR, incomboLR)

#%% look at synapse counts, ipsi vs contra for each MN in subgrp
#make new data frame with "muscle group" "num syns" and "side"
def pop_df(groupnames,inputmat,indsmats,typea):
    n = 0
    bimat = list()
    #indiv_muscs = ['1','2','3','4','5','6','7','8','9','10','11','12']*len(groupnames)
    if typea == 1:
        for i in np.arange(len(inputmat)):
            bimat.append(np.where(inputmat[i] > 0, 1, 0))
    elif typea == 2:
        bimat = inputmat
        
    synct = pd.DataFrame(columns = ['Num_Inputs','Muscle Group','Side'])
    statstbl = np.zeros([len(groupnames),2])
    
    for g in np.arange(0,len(groupnames)):
        #ipsicontramat = np.array([2,len(indsmats[g])])
        #make an if where if on the DL loop, Muscle group = DL for whole section at a time?; and for ipsi v contra (0,3 vs. 1,2), mean of vales set "side" accordingly
        mgrp = groupnames[g]
        
        ipsi = np.zeros(len(indsmats[g]))
        contra = np.zeros(len(indsmats[g]))
        #calculate the vals for each muscle in an np array then put in data frame below
        for m in np.arange(0,len(indsmats[g])):
            if typea == 1:
                ipsi[m] = ((np.sum(bimat[0][:,indsmats[g][m]]) + (np.sum(bimat[3][:,indsmats[g][m]]))))
                contra[m] = ((np.sum(bimat[1][:,indsmats[g][m]]) + (np.sum(bimat[2][:,indsmats[g][m]]))))
                synct.loc[str(n*2)] = [ipsi[m], mgrp, 'ipsi']
                synct.loc[str(n*2 + 1)] = [contra[m], mgrp, 'contra']
            elif typea == 2:
                ipsi[m] = np.mean([np.mean(bimat[0][:,indsmats[g][m]]), np.mean(bimat[3][:,indsmats[g][m]])])
                contra[m] = np.mean([np.mean(bimat[1][:,indsmats[g][m]]), np.mean(bimat[2][:,indsmats[g][m]])])
                synct.loc[str(n*2)] = [ipsi[m], mgrp, 'ipsi']
                synct.loc[str(n*2 + 1)] = [contra[m], mgrp, 'contra']
            n = n+1
       # ipsicontramat = np.reshape(ipsicontramat,(-1,1))
        
    #synct = synct.append()
        statstbl[g,:] = stats.mannwhitneyu(ipsi,contra,alternative = "two-sided")
    
    return synct, statstbl

def swarm_synct(synct,ti,typea):
    if typea == 1:
        yl = "Number of Inputs"
    elif typea == 2:
        yl = "Mean Weight of Inputs"
    f,ax = plt.subplots()
    sb.swarmplot(data = synct, x = "Muscle Group", y = "Num_Inputs", hue = "Side", palette = "rocket", dodge=True)
    ax.set(xlabel="Muscle Groups", ylabel= yl,title = ti)
    plt.savefig(ti + ".svg")
    
#%%
allct, allstatsw = pop_df(groupnames, allcomboLR, [dlinds, doinds, vlinds, ltinds, vainds, voinds],2)
swarm_synct(allct, 'Unilateral vs. Bilateral PMN-MN Inputs by Muscle Groups',2)

allct, allstatsc = pop_df(groupnames, allcomboLR, [dlinds, doinds, vlinds, ltinds, vainds, voinds],1)
swarm_synct(allct, 'Unilateral vs. Bilateral PMN-MN Inputs by Muscle Groups',1)

exct, exstatsw = pop_df(groupnames, excomboLR, [dlinds, doinds, vlinds, ltinds, vainds, voinds],2)
swarm_synct(exct, 'Unilateral vs. Bilateral PMN-MN Inputs by Muscle Groups - Excitatory Drive',2)

exct, exstatsc = pop_df(groupnames, excomboLR, [dlinds, doinds, vlinds, ltinds, vainds, voinds],1)
swarm_synct(exct, 'Unilateral vs. Bilateral PMN-MN Inputs by Muscle Groups - Excitatory Drive',1)

inct, instatsw = pop_df(groupnames, incomboLR, [dlinds, doinds, vlinds, ltinds, vainds, voinds],2)
swarm_synct(inct, 'Unilateral vs. Bilateral PMN-MN Inputs by Muscle Groups - Inhibitory Drive',2)

inct, instatsc = pop_df(groupnames, incomboLR, [dlinds, doinds, vlinds, ltinds, vainds, voinds],1)
swarm_synct(inct, 'Unilateral vs. Bilateral PMN-MN Inputs by Muscle Groups - Inhibitory Drive',1)

#%% multiside cossim fxn
#run cossim and store ACS
def cossim_multside(compmats):
    acsmat = np.zeros([len(compmats),len(compmats),len(compmats[0])])#output 1 mat for each side comp
    #first compare same muscle grp across the LL, LR, RL, and RR sides-store as first element 4 diff mats
    #and then concatenate into a bigger mat at the end
    for n in np.arange(0,len(compmats)): #loop thru muscle subgrps
        for m in np.arange(0,len(compmats[n])): #loop thru 4 sides         
            selfcos = np.nanmean(1-(sp.distance.cdist(compmats[n][m].T, compmats[n][m].T, 'cosine')))
            acsmat[n,n,m] = selfcos
            
            j = n + 1 #compare with neighbors
            while j < len(compmats):
                othercos = np.nanmean(1-(sp.distance.cdist(compmats[n][m].T, compmats[j][m].T, 'cosine')))
                #othermat = 1 - sp.distance.cdist(compmats[m].T, compmats[j].T, 'cosine')
                acsmat[n,j,m] = othercos
                acsmat[j,n,m] = othercos
                j = j + 1
            
    toleftmat = np.concatenate((acsmat[:,:,0],acsmat[:,:,2]),axis = 0)
    torightmat = np.concatenate((acsmat[:,:,1],acsmat[:,:,3]),axis = 0)
    bigmat = np.concatenate((toleftmat,torightmat),axis=1)
    
    collmat = np.concatenate(((acsmat[:,:,0] + acsmat[:,:,2])/2,(acsmat[:,:,1] + acsmat[:,:,3])/2), axis = 1)
            
    return acsmat, bigmat, collmat

#%%calculate groupwise cosine similarities - real data and for loop shuff mats
acsall, comball, collall = cossim_multside([dlmatall,domatall,vlmatall,ltmatall,vamatall,vomatall])
acsex, combex, collex = cossim_multside([dlmatex,domatex,vlmatex,ltmatex,vamatex,vomatex])
acsin, combin, collin = cossim_multside([dlmatin,domatin,vlmatin,ltmatin,vamatin,vomatin])

#%%plot the cosine similarity matrices per muscle group
#cosine similarity matrices
def plot_cos_mult(cosmat1,cosmat2,muscs,ti):
    f,ax = plt.subplots()
    ax1 = plt.subplot(1,2,1)
    ax1 = sb.heatmap(cosmat1, vmin=0, vmax=1)
    ax1.set(xlabel="Muscles D-->V", ylabel="Muscles (V --> D)",xticks=np.arange(len(muscs)), xticklabels = muscs,  yticks=np.arange(len(muscs)), yticklabels = muscs,title = 'Ipsilateral')
    for tick in ax1.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    for tick in ax1.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
        
    ax2 = plt.subplot(1,2,2)
    ax2 = sb.heatmap(cosmat2, vmin=0, vmax=1)
    ax2.set(xlabel="Muscles D-->V", ylabel="Muscles (V --> D)",xticks=np.arange(len(muscs)), xticklabels = muscs, yticks=np.arange(len(muscs)), yticklabels = muscs,title = 'Contralateral')
    for tick in ax2.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    for tick in ax2.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
        
    plt.suptitle(ti)
    plt.show()

#%%
groupnames = ['DL','DO','VL','LT','VA','VO']
plot_cos_mult(collall[:,0:6], collall[:,6:12], groupnames,'Cosine Similarity Across Muscle Groups- All PMNs')
plot_cos_mult(collex[:,0:6], collex[:,6:12], groupnames,'Cosine Similarity Across Muscle Groups - Excitatory PMNs')
plot_cos_mult(collin[:,0:6], collin[:,6:12], groupnames,'Cosine Similarity Across Muscle Groups - Inhibitory PMNs')

#%%generate shufmats for comparisons and pull out subsets of muscle groups
#shuffle weights matrix 1000x, choose PMN partners based on prob of MN input
def shufmat(cnxns,num_reps):
    rand_mats = [] #set this up to be a 1000d array; store each, then perform the wavg on each -- will extract mean xj or even plot all light then do avg dark; same with vars
    bicnxns = np.where(cnxns > 0, 1, 0)
    outputPMNs = np.sum(bicnxns,1) #how many cnxns each PMN makes
    totalMNin = np.sum(np.sum(bicnxns,0),0) #how many inputs each MN receives
    inputMNs = (np.sum(bicnxns,0)) / totalMNin #probability of input to each MN
    
    P = bicnxns.shape[0] 
    M = bicnxns.shape[1]
    
    Wshuf = np.zeros([P,M])
    
    for rep in range(num_reps):
        for pout in range(P):
            outputs = np.random.choice(M, outputPMNs[pout], replace=False, p=inputMNs)
            Wshuf[pout,outputs] = 1
        rand_mats.append(Wshuf)
        Wshuf = np.zeros([P,M])
        
    return rand_mats

test = shufmat(LLallfxbi,1000)

randallLR = [shufmat(LLallfxbi,1000), shufmat(LRallfxbi,1000), shufmat(RLallfxbi,1000), shufmat(RRallfxbi,1000)]
randexLR = [shufmat(LLexfxbi,1000), shufmat(LRexfxbi,1000), shufmat(RLexfxbi,1000), shufmat(RRexfxbi,1000)]
randinLR = [shufmat(LLinfxbi,1000), shufmat(LRinfxbi,1000), shufmat(RLinfxbi,1000), shufmat(RRinfxbi,1000)]

#%%pull out the connectivity for the shuffs
def shufsub_cossim(inds,matin):   
    acsmat = np.zeros([len(inds),len(inds),len(matin),len(matin[0])])
    acscombmat = np.zeros([len(inds),len(inds),2,len(matin[0])])
    
    for l in np.arange(0,len(matin[0])): #1000shuffles
        for n in np.arange(0,len(inds)): #musc subgrps     
            for m in np.arange(0,len(matin)): #4 sides
                selfcos = np.nanmean(1 - sp.distance.cdist(matin[m][l][:,inds[n]].T, matin[m][l][:,inds[n]].T, 'cosine'))
                acsmat[n,n,m,l] = selfcos
                
                j = n + 1
                while j < len(inds):
                    othercos = np.nanmean(1 - sp.distance.cdist(matin[m][l][:,inds[n]].T, matin[m][l][:,inds[j]].T, 'cosine'))
                    acsmat[n,j,m,l] = othercos
                    acsmat[j,n,m,l] = othercos
                    j = j + 1
        
        acscombmat[:,:,0,l] = (acsmat[:,:,0,l] + acsmat[:,:,2,l])/2
        acscombmat[:,:,1,l] = (acsmat[:,:,1,l] + acsmat[:,:,3,l])/2
    
                
    return acscombmat #make this a 5,5,2,1000 array - w/in,w/o x w/in,w/o x 1000


acsshufallcoll = shufsub_cossim([dlinds,d2inds,vlinds,ltinds,vainds,voinds], randallLR)
acsshufexcoll = shufsub_cossim([dlinds,d2inds,vlinds,ltinds,vainds,voinds], randexLR)
acsshufincoll = shufsub_cossim([dlinds,d2inds,vlinds,ltinds,vainds,voinds], randinLR)

            

dlshufall, dlshufex, dlshufin = findsubsetsshuf(dlinds)
doshufall, doshufex, doshufin = findsubsetsshuf(d2inds)
vlshufall, vlshufex, vlshufin = findsubsetsshuf(vlinds)
ltshufall, ltshufex, ltshufin = findsubsetsshuf(ltinds)
vashufall, vashufex, vashufin = findsubsetsshuf(vainds)
voshufall, voshufex, voshufin = findsubsetsshuf(voinds)

#%%plot the cosine similarity matrices per muscle group of example shuffle
groupnames = ['DL','DO','VL','LT','VA','VO']
plot_cos_mult(acsshufallcoll[:,:,0,24],acsshufallcoll[:,:,1,24], groupnames,'Cosine Similarity Across Muscle Groups- Shuffle All')
plot_cos_mult(acsshufexcoll[:,:,0,24],acsshufexcoll[:,:,1,24], groupnames,'Cosine Similarity Across Muscle Groups - Shuffle Excitatory')
plot_cos_mult(acsshufincoll[:,:,0,24],acsshufincoll[:,:,1,24], groupnames,'Cosine Similarity Across Muscle Groups - Shuffle Inhibitory')

#%% plot histograms for the ACS shufs vs real
def plot_cossim_dists(cosreal, cosshuf, title, groupnames):
    cosr1 = cosreal[:,0:6]
    cosr2 = cosreal[:,6:12]
    cosreal2d = [cosr1, cosr2]
    for m in np.arange(0,len(cosreal2d)):
        if m == 0:
            cat = ' - Ipsilateral'
        else:
            cat = ' - Contralateral'
        for i in np.arange(0,len(cosreal2d[0])):
            j = i
            while j < len(cosreal2d[0]):
                tiplot = title + cat + ': ' + groupnames[i] + ' x ' + groupnames[j]
                #get distrib of ACS
                dist = cosshuf[i,j,m,:]
            
                #find 95% confidence
                ci = stats.t.interval(alpha=0.95, df=len(dist)-1, loc=np.mean(dist), scale=np.std(dist))
                perc95 = ci[1]
                perc5 = ci[0]
         
                #plot it
                f,ax = plt.subplots()
                plt.hist(dist, bins = 50, color = 'c', alpha = 0.6)
                plt.axvline(cosreal2d[m][i,j], color = 'm', linewidth = 2)
                plt.axvline(perc95, color = 'k', linestyle = 'dashed', linewidth = 1)
                plt.axvline(perc5, color = 'k', linestyle = 'dashed', linewidth = 1)
                ax.set(xlabel="Cosine Similarity", ylabel="Frequency",title = tiplot)
        
                plt.tight_layout()
                plt.show()
                
                #next iter for this grp
                j = j + 1

#%%
grpnames = ['DL','DO','VL','LT','VA','VO']

plot_cossim_dists(collall,acsshufallcoll,'All PMNs',grpnames)
plot_cossim_dists(collex,acsshufexcoll,'Excitatory PMNs',grpnames)
plot_cossim_dists(collin,acsshufincoll,'Inhibitory PMNs',grpnames)


#LTs also have huge overlap from these plots; when check raw Jpm mat, looks like they should be sep
#also look to be sorted correctly in the LLallfx, LRallfx etc arrays

#2 thoughts
#1 - it could be that I need to do the cossim on the real vals. 
#and therefore set the shuf vals within some range matching gnereal distribtuion of weight magnitudes
#2 - checked that 4 mats coming out, and see that LTs are strongest in LR and RL mats?? possible that mats are switched? or psosible that LTs get contra only input (doesn't get better if swap mats for the mean)
#%%
#%%
#%%
#%%
#%%

#%% OLD PORTIONS OF THE CODE

#OLD SORTING
# #%% DLV grouping
# dor = ['1','9','2','10']
# lat = ['11','19','3','18','20','4','24','23','22','21','8','5']
# ven = ['12','13','30','14','6','7','25','26','27','29','28','15','16','17']
# allmuscs = dor + lat + ven

# #%%
# #for each PMN (row) (look at that row and the row below, idx by 2 so that we're seeing L & R copies of PMNs)
# epind = 0
# ipind = 0

# allPMNs = np.zeros([int(Jpm.shape[0]/2), len(allmuscs)])
# ePMNs = np.zeros([len(ex_pnames), len(allmuscs)])
# iPMNs = np.zeros([len(in_pnames), len(allmuscs)])

# for pindie in np.arange(0,len(Jpm),2):
#     #take sum of the two PMN-MN rows for each column
#     lrpm = np.sum(np.array(Jpm.iloc[pindie:pindie+2,1:]),0).T
#     for mi in np.arange(1,Jpm.shape[1],2):
#         mind = list()
#         #then take average of every two columns (L & R MNs) for each LR PMN pair
#         pm = np.mean(lrpm[mi:mi+2])
#         #for each MN name, check if number immediately preceded by 'MN' and suc by ' ' OR '/' (to check single vs. double digits)
#         mtemp = Jpm.columns[mi+1]
#         mtemp = mtemp.split('MN')[1]
#         mtemp = mtemp[:mtemp.index(" ")]
#         if '-' in mtemp:
#             mtemp = mtemp[:mtemp.index("-")]
#         #if '/' = 2 MNs, then store for the number strings before and after the '/'
#         if '/' in mtemp:
#             mtemp = mtemp.split('/')
#             for im,mn in enumerate(mtemp):
#                 mind.append(allmuscs.index(mtemp[im])) #find idx of this mn in the allmuscs list
#         else:
#              mind = allmuscs.index(mtemp)   
#         #store in PMN row MN col
#         pind = int(pindie/2)
#         allPMNs[pind,mind] = pm
#         #print('a_' + str(pindie) + '_' + str(pind) + ',' + str(mind))
        
#         #break into E vs. I matrices
#         if NTvals[int(pindie/2)] == 1:
#          #   print('e_' + str(pindie) + '_' + str(epind) + ',' + str(mind))
#             ePMNs[epind,mind] = pm
#             if mind == 18:
#                 epind = epind + 1
            
#         elif NTvals[int(pindie/2)] == -1:
#           #  print('i_' + str(pindie) + '_' + str(ipind) + ',' + str(mind))
#             iPMNs[ipind,mind] = pm
#             if mind == 18:
#                 ipind = ipind + 1
            

# #%% PMN DLV group plots
# #plot grouped weights matrices
# muscles = np.arange(0,30)
# plot_p2m_weights(allPMNs,muscles,allmuscs,pnames, 'All PMNs - spatial muscle order')
# plot_p2m_weights(ePMNs,muscles,allmuscs,ex_pnames, 'Excitatory PMNs - spatial muscle order')
# plot_p2m_weights(iPMNs,muscles,allmuscs,in_pnames, 'Inhibitory PMNs - spatial muscle order')

# #%%
# xjall, sigjall, reordall, zreall = wavg(allPMNs, muscles)
# xjex, sigjex, reordex, zreex  = wavg(ePMNs, muscles)
# xjin, sigjin, reordin, zrein = wavg(iPMNs, muscles)

# #%%
# #plot the sorted PMNs
# plot_p2m_weights(reordall,muscles,allmuscs,pnames, 'All PMNs - reordered by weighted average')
# plot_p2m_weights(zreall,muscles,allmuscs,pnames, 'All PMNs - reordered by weighted average, z-score')

# plot_p2m_weights(reordex,muscles,allmuscs,ex_pnames, 'Excitatory PMNs - reordered by weighted average')
# plot_p2m_weights(zreex,muscles,allmuscs,ex_pnames, 'Excitatory PMNs - reordered by weighted average, z-score')

# plot_p2m_weights(reordin,muscles,allmuscs,in_pnames, 'Inhibitory PMNs - reordered by weighted average')
# plot_p2m_weights(zrein,muscles,allmuscs,in_pnames, 'Inhibitory PMNs - reordered by weighted average, z-score')

#%%
#do it again for the locations being spatially clustered?
# musclocs = np.flip(np.array([14, 14, 13, 13, 12.5, 12, 11, 11, 11, 10.5,
#                      10, 9, 9, 9, 8, 8, 6, 5.5, 5, 5, 4.5, 4.5,
#                      4, 3.5, 3.5, 3, 2.5, 2, 1.5, 1]));

# wavgall, centroids, reordall, zreall, xjalllocs, sigjalllocs = wavg(allPMNs, musclocs)
# wavgex, centex, reordex, zreex, xjexlocs, sigjexlocs = wavg(ePMNs, musclocs)
# wavgin, centin, reordin, zrein, xjinlocs, sigjinlocs = wavg(iPMNs, musclocs)

#did not make substantial difference - do not pursue

#%%
#plot the sorted PMNs
# plot_p2m_weights(reordall,muscles,allmuscs,pnames)
# plot_p2m_weights(zreall,muscles,allmuscs,pnames)

# plot_p2m_weights(reordex,muscles,allmuscs,ex_pnames)
# plot_p2m_weights(zreex,muscles,allmuscs,ex_pnames)

# plot_p2m_weights(reordin,muscles,allmuscs,in_pnames)
# plot_p2m_weights(zrein,muscles,allmuscs,in_pnames)

#%%plot PMNs vs. wavg
#first sort the wavg by order
# plot_xbar(xjall, pnames,'All PMNs - weighted avg distrib')
# plot_xbar(xjex, pnames, 'Excitatory PMNs - weighted avg distrib')
# plot_xbar(xjin, pnames, 'Inhibitory PMNs - weighted avg distrib')

# # plot_xbar(xjalllocs, pnames)
# # plot_xbar(xjexlocs, pnames)
# # plot_xbar(xjinlocs, pnames)
# #%%
# #compare var's
# # msigma_all = np.mean(varshufall, axis = 1)
# # msigma_ex = np.mean(varshufex, axis = 1)
# # msigma_in = np.mean(varshufin, axis = 1)

# msigma_all = varshufall[:,20]
# msigma_ex = varshufex[:,20]
# msigma_in = varshufin[:,20]

# allvall = np.column_stack([sigjall, msigma_all])
# eve = np.column_stack([sigjex, msigma_ex])
# ivi = np.column_stack([sigjin, msigma_in])

# plot_var(allvall,['All PMNs','Shuf All'])
# plot_var(eve,['Excit PMNs','Shuf Excit'])
# plot_var(ivi,['Inhib PMNs','Shuf Inhib'])
    
# #%% compare the variances of real PMN-MN spread vs. shuffled w/ t-test
# tallvshufall = stats.ttest_ind(sigjall, msigma_all)
# texvshufex = stats.ttest_ind(sigjex, msigma_ex)
# tinvshufin = stats.ttest_ind(sigjin, msigma_in)

# #shows that indiv PMN spread is sig more spatially localized in connectome data than shuffled data

# #%% calculate the norm of D, L, and V matrices for connectome v shuffled data as a metric for overlap
# #if product of norms = 0, then orthogonal, no overlap; if > 0, non-orthogonal
# dorfxinds = np.arange(0,len(dorfx))
# latfxinds = np.arange(len(dorfx),len(dorfx)+len(latfx))
# venfxinds = np.arange(len(dorfx)+len(latfx),len(dorfx)+len(latfx)+len(venfx))

# #despite rerord matrices looking similar, group this matrix comparison according to my new spatial + fxnal muscle groups --> dorfx, latfx, venfx
# dorall = biallfx[:,dorfxinds]
# tdorall = dorall.T
# latall = biallfx[:,latfxinds]

# testdl = tdorall @ latall
# normtest = np.linalg.norm(testdl)
# normtestdor = np.linalg.norm(tdorall)
# normtestlat = np.linalg.norm(latall)

# venall = biallfx[:,venfxinds]

# norm_all_dor = np.linalg.norm(dorall)
# norm_all_lat = np.linalg.norm(latall)
# norm_all_ven = np.linalg.norm(venall)

# #%%
# #this makes no sense b/c of course these would be positive values? try instead
# #do dot product of row vectors for PMNs dor, lat, ven
# for td in tdorall:
#     testdot = np.dot(tdorall[td,:],latall[:,td])
# checkorthog = np.where(testdot==0)

# #do norm of row and col, then take product, sum all?
# dotdl = []
# for td in tdorall:
#     testnormd = np.linalg.norm(tdorall[td,:])
#     testnorml = np.linalg.norm(latall[:,td])
#     dotdl[td] = testnormd*testnorml

#another idea: try taking eigenvecs of each matrix and see if orthog
#can't do with non-square matrix and can't square this?
# eigdor = np.linalg.eig(dorall)
# eiglat = np.linalg.eig(latall)
# eigven = np.linalg.eig(venall)

# #%%
# #option: take average cosine similarity of groups DL to LL, etc. and plot distrib of this compared to the shuffled 1000x distrib to see if significant
# #redo into smaller subgrps based on first plot by eye
# # D1 (1, 9, 2, 10, 3, 11)
# # D2L1 (19, 20, 18, 24, 23, 22, 21, 8)
# # V2 (25, 26, 27, 29)
# # L2V1 (4, 5, 12, 13, 30, 14)
# # V3 (6, 7, 28, 15, 16, 17)

# #previous 
# dorfx = ['1','9','2','10','3','11','19','20']
# latfx = ['18','24','23','22','21','8','25','26','27','29']
# venfx = ['4','5','12','13','30','14','6','7','28','15','16','17']



# #%% shuffle matrices generation for all, e and i
# randall = shufmat(allPMNs,1000)
# randex = shufmat(ePMNs,1000)
# randin = shufmat(iPMNs,1000)

# #%%
# #run the wavg fxn on the shuf mats
# xjshufall = np.zeros([len(allPMNs),len(randall)])
# xjshufex = np.zeros([len(ePMNs),len(randall)])
# xjshufin = np.zeros([len(iPMNs),len(randall)])

# varshufall = np.zeros([len(allPMNs),len(randall)])
# varshufex = np.zeros([len(ePMNs),len(randall)])
# varshufin = np.zeros([len(iPMNs),len(randall)])

# matall = np.zeros([allPMNs.shape[0],allPMNs.shape[1],len(randall)])
# matex = np.zeros([ePMNs.shape[0], ePMNs.shape[1],len(randall)])
# matin = np.zeros([iPMNs.shape[0], iPMNs.shape[1],len(randall)])

# zmatall = np.zeros([allPMNs.shape[0],allPMNs.shape[1],len(randall)])
# zmatex = np.zeros([ePMNs.shape[0], ePMNs.shape[1],len(randall)])
# zmatin = np.zeros([iPMNs.shape[0], iPMNs.shape[1],len(randall)])

# for dim in np.arange(0,len(randall)):
#     xjshufall[:,dim], varshufall[:,dim], matall[:,:,dim], zmatall[:,:,dim] = wavg(randall[dim], muscles)
#     xjshufex[:,dim], varshufex[:,dim], matex[:,:,dim], zmatex[:,:,dim] = wavg(randex[dim], muscles)
#     xjshufin[:,dim], varshufin[:,dim], matin[:,:,dim], zmatin[:,:,dim] = wavg(randin[dim], muscles)

# #%% plot some of the shuffled matrices and see how they compare
# pickrall = np.random.randint(0, matall.shape[2], size = 20)
# pickrex = np.random.randint(0, matex.shape[2], size = 20)
# pickrin = np.random.randint(0, matin.shape[2], size = 20)

# # for i in np.arange(0,len(pickrall)):
# #     plot_p2m_weights(matall[:,:,pickrall[i]],muscles,allmuscs,pnames,'PMN-MN Shuffled Weights - All - matrix '+ str(pickrall[i]))
# #     plot_p2m_weights(matex[:,:,pickrex[i]],muscles,allmuscs,ex_pnames,'PMN-MN Shuffled Weights - Excitatory matrix'+ str(pickrex[i]))
# #     plot_p2m_weights(matin[:,:,pickrin[i]],muscles,allmuscs,in_pnames,'PMN-MN Shuffled Weights - Inhibitory matrix'+ str(pickrin[i]))

# #%% compare binary plots
# biall = np.where(reordall>0, 1, 0)
# biex = np.where(reordex>0, 1, 0)
# biin = np.where(reordin>0, 1, 0)

# plot_p2m_weights(biall,muscles,allmuscs,pnames,'PMN-MN Connectome Weights - All')
# plot_p2m_weights(biex,muscles,allmuscs,ex_pnames,'PMN-MN Connectome Weights - Excitatory')
# plot_p2m_weights(biin,muscles,allmuscs,in_pnames,'PMN-MN Shuffled Weights - Inhibitory')
# # #%%
# # #compare xj's
# mxj_all = np.mean(xjshufall, axis = 1)
# mxj_ex = np.mean(xjshufex, axis = 1)
# mxj_in = np.mean(xjshufin, axis = 1)

# plot_xbar(xjall, mxj_all, pnames)
# plot_xbar(xjex, mxj_ex, pnames)
# plot_xbar(xjin, mxj_in, pnames)

# #%% compare xj's across 20x shuf vs connectome
# plot_xbar(xjall, xjshufall[:,pickrall], "All PMNs")
# plot_xbar(xjex, xjshufex[:,pickrex], "Excitatory PMNs")
# plot_xbar(xjin, xjshufin[:,pickrin], "Inhibitory PMNs")


#%%

# #define f test
# def f_test(group1, group2):
#     f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
#     nun = group1.size-1
#     dun = group2.size-1
#     p_value = 1-stats.f.cdf(f, nun, dun)
#     return f, p_value
#%%
#regress and check if population variances of samples are equal, ftest for variances of populations
# rall,mrall = regressline(xjallfx)
# rshufall,mrshufall = regressline(mxj_all)
# varallvshuf = levene(mrall.resid, mrshufall.resid)
# fall = f_test(xjallfx,mxj_all)

# rex,mrex = regressline(xjexfx)
# rshufex,mrshufex = regressline(mxj_ex)
# varexvshuf = levene(mrex.resid, mrshufex.resid)
# fex = f_test(xjexfx,mxj_ex)

# rin,mrin = regressline(xjinfx)
# rshufin,mrshufin = regressline(mxj_in)
# varinvshuf = levene(mrin.resid, mrshufin.resid)
# fin = f_test(xjinfx,mxj_in)

#plot the data plus the regression lines
# plot_xbar(xjallfx, xjshufall[:,pickrall], rall, rshufall, "All PMNs")
# plot_xbar(xjexfx, xjshufex[:,pickrex], rex, rshufex, "Excitatory PMNs")
# plot_xbar(xjinfx, xjshufin[:,pickrin], rin, rshufin, "Inhibitory PMNs")

#%%
#compare var's
# msigma_all = np.mean(varshufall, axis = 1)
# msigma_ex = np.mean(varshufex, axis = 1)
# msigma_in = np.mean(varshufin, axis = 1)

# msigma_all = varshufall[:,20]
# msigma_ex = varshufex[:,20]
# msigma_in = varshufin[:,20]

# allvall = np.column_stack([sigjall, msigma_all])
# eve = np.column_stack([sigjex, msigma_ex])
# ivi = np.column_stack([sigjin, msigma_in])

# plot_var(allvall,['All PMNs','Shuf All'])
# plot_var(eve,['Excit PMNs','Shuf Excit'])
# plot_var(ivi,['Inhib PMNs','Shuf Inhib'])
    
# #%% compare the variances of real PMN-MN spread vs. shuffled w/ t-test
# tallvshufall = stats.ttest_ind(sigjall, msigma_all)
# texvshufex = stats.ttest_ind(sigjex, msigma_ex)
# tinvshufin = stats.ttest_ind(sigjin, msigma_in)

#shows that indiv PMN spread is sig more spatially localized in connectome data than shuffled data

#%% not using anymore


# # #indiv circumferential connectivity check
# # muscles = np.arange(0,30)

# # #plot each PMN's connectivity distribution profile
# # for pi,pr in enumerate(allPMNs):
# #     plot_musc_syndist(pi,muscles,allmuscs,pnames)

# List generation and heatmap of PMN DLV groups
#function for simplified weights DLV
# def dlvsum(mat_in,dorinds,latinds,veninds):
#     mat_out = np.zeros([len(mat_in),3])
#     for i,p in enumerate(mat_in):
#         dorPMNs = sum(mat_in[i,dorinds])
#         latPMNs = sum(mat_in[i,latinds])
#         venPMNs = sum(mat_in[i,veninds])
#         mat_out[i,:] = [dorPMNs, latPMNs, venPMNs]
        
#     return mat_out

# #go through allPMNs and sum weights to dor, lat, and ven indices, and generate lists
# gnames = ['dorsal','lateral','ventral']
# dorinds = np.arange(0, len(dor))
# latinds = np.arange(len(dor), len(dor)+len(lat))
# veninds = np.arange(len(dor)+len(lat), len(dor)+len(lat)+len(ven))

# all_sum = dlvsum(allPMNs,dorinds,latinds,veninds)
# e_sum = dlvsum(ePMNs,dorinds,latinds,veninds)
# i_sum = dlvsum(iPMNs,dorinds,latinds,veninds)

# #plot the summed inputs of each PMN to D,L,V groups
# plot_p2dlv_weights(all_sum,gnames,pnames)
# plot_p2dlv_weights(e_sum,gnames,ex_pnames)
# plot_p2dlv_weights(i_sum,gnames,in_pnames)


#%% REDO the LR syn count and syn weight comparisons by working with Aref's presorted srpeadsheets
#make new data frame with "muscle group" "num syns" and "side" from presorted data
grpfiles = ["DL R L PMNs", "DOLOLL", "VL RL PMNs", "T R L PMNs", "SNc RL PMNs", "VO R L PMNs"]

DLsort = pd.read_excel('PMNs R L to MNs R L normalized v5_arefshare_1222.xlsx', sheet_name=grpfiles[0])
DOsort = pd.read_excel('PMNs R L to MNs R L normalized v5_arefshare_1222.xlsx', sheet_name=grpfiles[1])
VLsort = pd.read_excel('PMNs R L to MNs R L normalized v5_arefshare_1222.xlsx', sheet_name=grpfiles[2])
LTsort = pd.read_excel('PMNs R L to MNs R L normalized v5_arefshare_1222.xlsx', sheet_name=grpfiles[3])
VAsort = pd.read_excel('PMNs R L to MNs R L normalized v5_arefshare_1222.xlsx', sheet_name=grpfiles[4])
VOsort = pd.read_excel('PMNs R L to MNs R L normalized v5_arefshare_1222.xlsx', sheet_name=grpfiles[5])

#%%
def pop_df_presort(groupnames,inputmat,typea): 
    synct = pd.DataFrame(columns = ['Num_Inputs','Muscle Group','Side'])
    statstbl = np.zeros([len(groupnames),2])

    #pos = 0
    bimat = list()
    wmat = list()
    #indiv_muscs = ['1','2','3','4','5','6','7','8','9','10','11','12']*len(groupnames)
    if typea == 1:
        for i in np.arange(len(inputmat)):
            for j in np.arange(len(inputmat[i])):
                bimat.append(np.where(inputmat[i][j] > 0, 1, 0))
    elif typea == 2:
        for i in np.arange(len(inputmat)):
            for j in np.arange(len(inputmat[i])):
                wmat.append(np.where(inputmat[i][j]==0,np.nan,inputmat[i][j]))
    elif typea == 3:
        for i in np.arange(len(inputmat)):
            for j in np.arange(len(inputmat[i])):
                wmat.append(np.where(inputmat[i][j]==0,np.nan,inputmat[i][j]))
                bimat.append(np.where(inputmat[i][j] > 0, 1, 0))
    n = 0
    #now go by the number of inds in the group but thru all dims of the inputmat structure
    for g in np.arange(0,len(groupnames)):

        #ipsicontramat = np.array([2,len(indsmats[g])])
        #make an if where if on the DL loop, Muscle group = DL for whole section at a time?; and for ipsi v contra (0,3 vs. 1,2), mean of vales set "side" accordingly
        mgrp = groupnames[g]   
     
        #calculate the vals for each muscle in an np array then put in data frame below
        unilat = np.zeros([len(groupnames),wmat[g*4].shape[1]])
        bilat = np.zeros([len(groupnames),wmat[g*4].shape[1]])
        
        for m in np.arange(0,wmat[g*4].shape[1]):
            if typea == 1:
                unilat[g,m] = np.sum(bimat[int(g*4)+0][:,m]) + (np.sum(bimat[int(g*4)+3][:,m]))
                bilat[g,m] = np.sum(bimat[int(g*4)+1][:,m]) + (np.sum(bimat[int(g*4)+2][:,m]))
                synct.loc[str(n*2)] = [unilat[g,m], mgrp, 'unilateral']
                synct.loc[str(n*2 + 1)] = [bilat[g,m], mgrp, 'bilateral']
            elif typea == 2:
                unilat[g,m] = np.nanmean([np.nanmean(wmat[int(g*4)+0][:,m]), np.nanmean(wmat[int(g*4)+3][:,m])])
                bilat[g,m] = np.nanmean([np.nanmean(wmat[int(g*4)+1][:,m]), np.nanmean(wmat[int(g*4)+2])])
                synct.loc[str(n*2)] = [unilat[g,m], mgrp, 'unilateral']
                synct.loc[str(n*2 + 1)] = [bilat[g,m], mgrp, 'bilateral']
                
            #do weighted avg of inputs
            elif typea == 3: 
                unilat[g,m] = np.mean([np.average(bimat[int(g*4)+0][:,m]!=0,weights = wmat[int(g*4)+0][:,m]!=np.nan), np.average(bimat[int(g*4)+3][:,m]!=0, weights = wmat[int(g*4)+3][:,m]!=np.nan)])
                bilat[g,m] = np.mean([np.average(bimat[int(g*4)+1][:,m]!=0,weights = wmat[int(g*4)+1][:,m]!=np.nan), np.average(bimat[int(g*4)+2][:,m]!=0, weights = wmat[int(g*4)+2][:,m]!=np.nan)])
                synct.loc[str(n*2)] = [unilat[g,m], mgrp, 'unilateral']
                synct.loc[str(n*2 + 1)] = [bilat[g,m], mgrp, 'bilateral']
            n = n+1
        
        unilat = unilat[~np.isnan(unilat)]
        bilat = bilat[~np.isnan(bilat)]
        statstbl[g,:] = stats.mannwhitneyu(unilat,bilat,alternative = "two-sided")
        # ipsicontramat = np.reshape(ipsicontramat,(-1,1))
    
#synct = synct.append()
    
    
    return synct, statstbl

def swarm_synct(synct,ti,typea):
    if typea == 1:
        yl = "Number of Inputs"
        sti = ti + '-numin'
    elif typea == 2:
        yl = "Mean Weight of Inputs"
        sti = ti + '-weights-nonzero'
    elif typea == 3:
        yl = "Weighted Average of Inputs"
        sti = ti + '-wavg-nonzero'
    f,ax = plt.subplots()
    sb.swarmplot(data = synct, x = "Muscle Group", y = "Num_Inputs", hue = "Side", palette = "rocket", dodge=True)
    ax.set(xlabel="Muscle Groups", ylabel= yl,title = ti)
    plt.savefig(sti + ".svg")
    
#%%
allgrps = [DLsort, DOsort, VLsort, LTsort, VAsort, VOsort]

arefmat = list()

for g in np.arange(len(allgrps)):
    nump = allgrps[g].shape[0]
    numm = allgrps[g].shape[1]-1
    inmat = [np.array(allgrps[g].iloc[0:int(nump/2),1:int(numm/2)+1]), np.array(allgrps[g].iloc[int(nump/2):nump+1,1:int(numm/2)+1]), 
             np.array(allgrps[g].iloc[0:int(nump/2),int(numm/2)+1:numm+1]), np.array(allgrps[g].iloc[int(nump/2):nump+1,int(numm/2)+1:numm+1])]
    arefmat.append(inmat)

#%%
fullsync, fullstats = pop_df_presort(groupnames,arefmat,1)
swarm_synct(fullsync,'Unilateral vs. Bilateral PMN-MN Inputs by Muscle Groups',1)

fullsyncw, fullstatsw = pop_df_presort(groupnames,arefmat,2)
swarm_synct(fullsyncw,'Unilateral vs. Bilateral PMN-MN Inputs by Muscle Groups',2)


fullsyncwavg, fullstatswwavg = pop_df_presort(groupnames,arefmat,3)
swarm_synct(fullsyncwavg,'Unilateral vs. Bilateral PMN-MN Inputs by Muscle Groups',3)
