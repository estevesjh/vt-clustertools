#!/usr/bin/env python -W ignore::DeprecationWarning
# gmmModule.py
# Purpose: Fit a gaussian mixture modelling on the color distribution of a given galaxy cluster catalog. 
## v2 - Study of the background Subtraction

# Description: In order to do the fit. First, it subtracts the background color distribution. 
# Second, produce a color distribution subtracted weighted by the membership probability.
# Third, it fit two gaussians, the red sequence and the blue cloud. 
# Finnaly, it produces the outputs that are the color probabilities (Pred,Pblue and Pcolor) and the gaussian parameter (mean_i,sigma_i,amplitude_i).

# input: color_data,mag_data,background_histograms,Pdist,cluster_ID,cluster_Z,galaxyID,hostID,r_cl,maxcol,maglim
# output: mu_r,mu_b,sigma_r,sigma_b,alpha_r,alpha_b,converged,Pred,Pblue,Pcolor,probgalid#,colfit,colweights,n_subtracted#,colfit2,colweights2,colfit3,colweights3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import mixture
from scipy.interpolate import interp1d

from matplotlib.ticker import NullFormatter
from matplotlib.colors import ListedColormap
# import seaborn as sns; sns.set()
import warnings
from scipy import interpolate
from scipy.stats import binned_statistic
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
import scipy.stats as stats
import matplotlib as mpl
import os
# mpl.style.use('seaborn-whitegrid')
mpl.style.use('seaborn-darkgrid')
import warnings
warnings.simplefilter('ignore', DeprecationWarning)

from matplotlib import cm
from matplotlib.colors import ListedColormap

top = cm.get_cmap('RdBu', 128)
midle = cm.get_cmap('Paired')
bottom = cm.get_cmap('RdBu_r', 128)
newcolors = np.vstack((top(np.linspace(0., 0.30, 128)),midle(np.linspace(0.2,0.3,16)),midle(np.linspace(0.3,0.2,16)),
                       bottom(np.linspace(0.30, 0., 128))))
newcmp = ListedColormap(newcolors, name='RdGrBu')

def colorLabel(N):
    colorLabel = [['$r$','$g-r$','$\Delta (g-r)$','color_gr'],['$i$','$r-i$','$\Delta (r-i)$','color_ri'],['$z$','$i-z$','$\Delta (i-z)$','color_iz']]
    return colorLabel[N]
def PDF_color(xvec,xmean,sigma):
    res = stats.norm.pdf(xvec,xmean,sigma)
    # norm = stats.norm.pdf(xmean,xmean,sigma)
    return res

def doBinColor(col,weight=None,x0=-1,xend=4.1,width=0.01,density=True):
    hist_bins=np.arange(x0,xend+width,width)
    col_hist,_=np.histogram(col,bins=hist_bins,weights=weight,density=density)
    center=(hist_bins[:-1]+hist_bins[1:])/2.
    width=np.diff(hist_bins)
    return col_hist, center, round(width[0],6)

def doSublistToArray(list1):
    vec=np.array([item for sublist in list1 for item in sublist])
    vec.shape=(vec.size,)
    return vec

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def writeParamLabel(param):
    mur,mub,sigr,sigb,alphar,alphab,conv = param
    line1='$\\sigma_r=%.2f \quad \\sigma_b=%.2f$ \n'%(sigr,sigb)
    line2='$\\mu_r=%.2f \quad \\mu_b=%.2f$ \n'%(mur,mub)
    line3='$\\alpha_r=%.2f \quad \\alpha_b=%.2f$ \n'%(alphar,alphab)
    res = line1+line2+line3
    return res

def IsRSmean(rs_mean,idx):
    if rs_mean is not None:
        expred=float(rs_mean[idx]) ## Mean RS Color
    else:
        expred=0
    return expred

# def IsValidCondition(ix):
#     # Is it non zero length?
#     sub_size = np.size(ix) - np.count_nonzero(ix)
#     return sub_size
    
def auxRS(rs_param,idx):
    expred_vec, slope_vec, intercept_vec = rs_param
    expred, slope, intercept = float(expred_vec[idx]),float(slope_vec[idx]),float(intercept_vec[idx])
    return expred,[slope, intercept]

def getScale(scale_vec,idx):
    if scale_vec is not None:
        scale_bg=scale_vec[idx]
    else:
        scale_bg = 1
    return scale_bg

def doRS_Color(mag,param):
    a, b = param
    rs_color = float(a)*mag+float(b)
    rs_color.shape = (len(mag),)
    return rs_color

def doRS_Offset(mag,color,param):
    a, b = param
    rs_color = float(a)*mag+float(b)
    offset = color-rs_color
    return offset

## Plot
def RS_plot_fit(color,mag,param,paramRS,Pred,Pblue,prob_weights=None,maglim=25,labely=r'$g-r$',labelx=r'$r$',Name="RM",zcls=0.1,save=None,hpc=0):
    if hpc:    
        savedir='/data/des61.a/data/johnny/clusters/test/plots/'#/redmapper_y1a1/localsub/test'
        # savedir='/data/des61.a/data/johnny/clusters/test/plots/MOF_withSub/'
        # savedir='/data/des61.a/data/johnny/clusters/test/plots/gsample/'#/redmapper_y1a1/localsub/test'
    else:
        savedir='./plots/'
    
    ## get the gaussian parameters
    mur,mub,sigr,sigb,alphar,alphab,_ = param
    
    ## Set up the figure
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.12]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(10, 8))

    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    axScatter = plt.axes(rect_scatter)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    ylmin,ylmax = (mub-0.75), (mur+0.75)
    xlmin,xlmax = (mag.mean()-3*mag.std()-1.2),(24.)

    blue, red = '#024794','#AF2B2F'

    ## get the RS vector
    slope, intercept = paramRS
    mag_vec = np.arange(xlmin,xlmax+0.01,0.01)
    RS = slope*mag_vec+intercept

    
    # ------> Do the scatter plot
    # Pred = np.array(Pred).reshape(len(color),1)
    # Pblue = np.array(Pblue).reshape(len(color),1)
    Pred = np.array(Pred)
    Pblue = np.array(Pblue)
    color_weights = Pred-Pblue
    # c = (color_weights == None)&(color_weights < -1)&(color_weights > 1)
    c = (color_weights == None)
    color_weights[c] = 0
    
    top = cm.get_cmap('RdBu_r', 128)
    midle = cm.get_cmap('Paired')
    bottom = cm.get_cmap('RdBu', 128)
    newcolors = np.vstack((top(np.linspace(0., 0.35, 128)),midle(np.linspace(0.2,0.3,8)),midle(np.linspace(0.3,0.2,8)),
                        bottom(np.linspace(0.35, 0., 128))))
    mycmp = ListedColormap(newcolors, name='RdGrBu')
    rdmp = ListedColormap(top(np.linspace(0.62, 1., 128)),name='Rd')
    # mycmap = ListedColormap(sns.color_palette("RdBu_r", 7).as_hex())
    axScatter.scatter(mag, color,c=color_weights,s=100*(prob_weights)**(1)+0.01,alpha=0.6,cmap=rdmp,label=r'$P_{mem}$')
    axScatter.plot(mag_vec,RS,color=red,linestyle='--',label='Best Fit RS')
    axScatter.axvline(maglim,linestyle='--',linewidth=2.,color=blue,label='$mag_{lim} = %.1f$'%(maglim))
    axScatter.set_xlabel(labelx)
    axScatter.set_ylabel(labely)
    axScatter.set_xlim((xlmin,xlmax))
    axScatter.set_ylim((ylmin,ylmax)) 
    #make a legend:
    legendSizes = [0.1,0.5, 1.0]
    for lS in legendSizes:
        plt.scatter([], [],s=100*(lS)**(1)+0.01,c='gray',alpha=0.6,label='$ %.1f$'%(lS))
    axScatter.legend()
    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h[0:], l[0:], borderpad=1,frameon=True, framealpha=0.3, edgecolor="k", facecolor="w")
    # ------> Do the vertical histogram
    yvec = np.linspace(ylmin,ylmax,1000)
    ybins = np.arange(ylmin, ylmax + 0.075, 0.075)
    width=np.diff(ybins)
    scale_r = np.sum(prob_weights*Pred*np.ones_like(color)*width[0]); scale_b = np.sum(prob_weights*Pblue*np.ones_like(color)*width[0])
    # fithist,fitbins=np.histogram(color,bins=ybins,weights=prob_weights,density=True)
    # norm = np.sum(fithist*width[0])
    axHisty.hist(color, bins=ybins, edgecolor='white',linewidth=1.2,align='mid',weights=prob_weights,color='lightgray',orientation='horizontal',alpha=1.0,histtype='bar',label=r'$P_{mem} \times N_{gal}$')
    axHisty.hist(color, bins=ybins, edgecolor='white',linewidth=1.2,align='mid',weights=prob_weights*Pred,color=red,orientation='horizontal',alpha=0.7,histtype='bar',label=r'$P_{red} \times N_{gal}$')
    axHisty.hist(color, bins=ybins, edgecolor='white',linewidth=1.2,align='mid',weights=prob_weights*Pblue,color=blue,orientation='horizontal',alpha=0.5,histtype='bar',label=r'$P_{blue} \times N_{gal}$')
    axHisty.plot(scale_r*stats.norm.pdf(yvec,mur,sigr)*alphar,yvec,color=red,linestyle="--")
    axHisty.plot(scale_r*stats.norm.pdf(yvec,mub,sigb)*alphab,yvec,color=blue,linestyle="--")
    axHisty.legend()
    # fithist,fitbins=np.histogram(color,bins=ybins,weights=prob_weights*Pred)
    # normr = np.sum(fithist*width[0])
    # fithist,fitbins=np.histogram(color,bins=ybins,weights=prob_weights*Pblue)
    # normb = np.sum(fithist*width[0])
    # Height = np.max(fithist)
    # color2, idx = np.sort(color), np.argsort(color)
    # axHisty.plot(norm*Pred[idx],color2,color=red,label=r'$P_{red} \times N_{gal}$')
    # axHisty.plot(norm*Pblue[idx],color2,color=blue,label=r'$P_{blue} \times N_{gal}$')
    
    # axHisty.plot(normr*stats.norm.pdf(yvec,mur,sigr),yvec,color=red,linestyle="--",label="Red Sequence")
    # axHisty.plot(normb*stats.norm.pdf(yvec,mub,sigb),yvec,color=blue,linestyle="--",label="Blue Cloud")
    
    # ------> Do the horizontal histogram
    binwidth = 0.15
    xbins = np.arange(xlmin,xlmax + 0.5, 0.5)
    axHistx.hist(mag, bins=xbins,weights=prob_weights, edgecolor='white',linewidth=1.2,align='mid',color=blue,alpha=0.8,histtype='bar')
    # axHistx.hist(mag, bins=xbins,weights=prob_weights*Pred,color='r',alpha=0.9,histtype='bar')
    # axHistx.hist(mag, bins=xbins,weights=prob_weights*Pblue,color='b',alpha=0.7,histtype='bar')
    axHistx.set_title(Name+' at z=%.3f'%(zcls))
    axHistx.legend()
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # h, l = plt.gca().get_legend_handles_labels()
    # # axScatter.legend(h[1:], l[1:], loc=0, labelspacing=0.5,frameon=True, framealpha=0.3, edgecolor="gray", facecolor="lightgray")
    # plt.legend(h[1:], l[1:], labelspacing=0.5,loc='upper left', borderpad=1,title=r'$P_{mem}$',bbox_to_anchor=(0.2,0.30,1.,1.),frameon=True, framealpha=0.3, edgecolor="gray")

    ## Save 
    if save is None:
        save=labely.split('$')[1]
    out_name = savedir+str(Name)+'_RS_Fit_%s.png'%(save)
    plt.savefig(out_name)
    # plt.show()
    plt.clf()


def doPlot_colorModel(color,Pred,Pblue,Pcolor,Pmem,param,kappa,zcl,maglim,labelx='color',name='RM',save=None,hpc=0):
    # Two subplots - Color Model:(Probability curves, Histograms)
    mur,mub,sigr,sigb,alphar,alphab,conv = param
    f, ax = plt.subplots(2, figsize=(8, 8), sharex=True)
    plt.subplots_adjust(hspace = .05)
    a,b = (mur-3*sigr-0.5), (mur+3*sigr+0.3)

    blue, red = 'lightblue', 'lightcoral'
    blue2, red2 = '#024794','#AF2B2F'

    ax[0].set_title('Color Model: %s at z = %.3f'%(name,zcl))
    ax[0].set_xlim(a,b)
    ax[1].set_xlim(a,b)

    color1, idx = np.sort(color), np.argsort(color)
    ## Probability curves: Pred, Pblue and Pcolor
    ax[0].plot(color1, Pred[idx],color=red2,label=r'$P_{red}$')
    ax[0].plot(color1,Pblue[idx],color=blue2,label=r'$P_{blue}$')
    ax[0].plot(color1,Pcolor[idx],color='lightgrey',linewidth=2.,linestyle='--',label=r'$P_{color}$')
    ax[0].set_ylabel('Probability')
    ax[0].legend(loc='upper right')
    ax[0].set_xlim(a-0.1,b+0.1)
    plt.figtext(0.15,0.87,'$mag_{lim} = %.1f \quad kappa = %.1f $ \n converged: %s '%(maglim,kappa,conv),ha='left',va='top')

    width=0.05
    yvec = np.linspace(a,b,1000); ybins = np.arange(a, b + width, width)
    scale_r = np.sum(Pmem*Pred*np.ones_like(color)*width); scale_b = np.sum(Pmem*Pblue*np.ones_like(color)*width)
    ## Histogram: Color distribution, two population red and blue.
    ax[1].hist(color, bins=ybins, edgecolor='white',linewidth=1.2,align='mid',weights=Pmem*Pred,color=red,alpha=0.9,histtype='bar',label=r'$P_{red} \; P_{mem}\; N_{gal}$')
    ax[1].hist(color, bins=ybins, edgecolor='white',linewidth=1.2,align='mid',weights=Pmem*Pblue,color=blue,alpha=0.9,histtype='bar',label=r'$P_{blue} \; P_{mem}\; N_{gal}$')
    ax[1].plot(yvec,scale_r*alphar*stats.norm.pdf(yvec,mur,sigr),color=red2,linestyle="--",label='Red Sequence Fit')
    ax[1].plot(yvec,scale_b*alphab*stats.norm.pdf(yvec,mub,sigb),color=blue2,linestyle="--",label='Blue Cloud Fit')
    ax[1].set_xlabel(labelx)
    paramLabel=writeParamLabel(param)
    plt.figtext(0.15,0.39,paramLabel,ha='left',va='bottom')
    ax[1].set_ylabel('member galaxies')
    ax[1].legend()
    ax[1].set_xlim(a-0.1,b+0.1)

    if hpc:
        savedir='/data/des61.a/data/johnny/clusters/test/plots/'#/redmapper_y1a1/localsub/test'
        # savedir='/data/des61.a/data/johnny/clusters/test/plots/MOF_withSub/'
        # savedir='/data/des61.a/data/johnny/clusters/test/plots/gsample/'
    else:
        savedir='./plots/'
    if save is None:
        save='color_'+labelx.split('$')[1]
    out_name = savedir+(name)+'_%s.png'%(save)
    plt.savefig(out_name)
    plt.close()

def doPlot_colorModel_bg(color,color_bg,Pred,Pblue,Pcolor,Pmem,scale,Pz_bg,param,kappa,zcl,maglim,mean,labelx='color',name='RM',save=None,hpc=0):
    
    # Two subplots - Color Model:(Probability curves, Histograms)
    mur,mub,sigr,sigb,alphar,alphab,conv = param
    mu_bg, sig_bg = weighted_avg_and_std(color_bg, Pz_bg)
    a,b = (mean-1.0), (mean+0.6)

    f, ax = plt.subplots(2, figsize=(8, 8), sharex=True)
    plt.subplots_adjust(hspace = .05)

    ax[0].set_title('Backgroun Subtraction: %s at z = %.3f'%(name,zcl))
    ax[0].set_xlim(a,b)
    ax[1].set_xlim(a,b)

    blue, red = 'lightblue', 'lightcoral'
    blue2, red2 = '#024794','#AF2B2F'

    width=0.05
    yvec = np.linspace(a,b,1000); ybins = np.arange(a, b + width, width)
    scale_w = np.sum(len(color)*width); scale_bg = np.sum(len(color_bg)*width*scale)
    
    hist, center, w = doBinColor(color,weight=Pmem,width=width,density=True)
    hist_bg, center, w = doBinColor(color_bg,weight=Pz_bg,width=width,density=True)
    hist_red, center, w = doBinColor(color,weight=Pmem*Pred,width=width)
    hist_blue, center, w = doBinColor(color,weight=Pmem*Pblue,width=width)

    hist_sub = (scale_w*hist-scale_bg*hist_bg); 
    hist_sub[hist_sub<0]=0
    kappa = np.sum(hist_sub*width)
    ## Histogram: Color distribution, two population red and blue.
    ax[1].bar(center,hist_sub,width=width,edgecolor='white',align='center',linewidth=1.2,color='lightgray',alpha=1.,label=r'$ P_{mem} \;N_{gal} \; - P_{z} \; N_{bkg}$')
    ax[1].plot(yvec,kappa*stats.norm.pdf(yvec,mur,sigr)*alphar,color=red2,linestyle="--")
    ax[1].plot(yvec,kappa*stats.norm.pdf(yvec,mub,sigb)*alphab,color=blue2,linestyle="--")
    ax[1].axvline(x=mean,linestyle='--',label='Expec. RS mean color')
    ax[1].legend()
    ax[1].set_xlabel(labelx)

    ax[0].bar(center,scale_w*hist, width=width,edgecolor='white',align='center',linewidth=1.2,color='gray',alpha=0.8,label=r'$ P_{mem} \; N_{gal}$')
    ax[0].bar(center,scale_bg*hist_bg,width=width,edgecolor='white',align='center',color=red,alpha=0.7,label=r'$ P_{z} \; N_{bkg})$')
    pdf_g = stats.norm.pdf(yvec, mu_bg, sig_bg) # now get theoretical values in our interval  
    ax[0].plot(yvec, scale_bg*pdf_g, color='gray',label="Bkg") # plot it
    ax[0].plot(yvec,scale_bg*pdf_g+kappa*stats.norm.pdf(yvec,mur,sigr)*alphar,color=red2,linestyle="--",label='RS+Bkg')
    ax[0].plot(yvec,scale_bg*pdf_g+kappa*stats.norm.pdf(yvec,mub,sigb)*alphab,color=blue2,linestyle="--",label='BC+Bkg')
    ax[0].axvline(x=mean,linestyle='--')
    
    ax[0].legend(loc='upper right')
    paramLabel=writeParamLabel(param)
    paramLabel=paramLabel+'$mag_{lim} = %.1f$ \n converged: %s '%(maglim,conv)
    plt.figtext(0.15,0.75,paramLabel,ha='left',va='bottom')
    ax[0].set_ylabel('N')

    if hpc:
        savedir='/data/des61.a/data/johnny/clusters/test/plots/'#/redmapper_y1a1/localsub/test'
        # savedir='/data/des61.a/data/johnny/clusters/test/plots/MOF_withSub/'
        # savedir='/data/des61.a/data/johnny/clusters/test/plots/gsample/'
    else:
        savedir='./plots/'
    if save is None:
        save='color_'+labelx.split('$')[1]
    out_name = savedir+(name)+'_%s_bgsub.png'%(save)
    plt.savefig(out_name)
    plt.close()

def DoClip(data,mean,sigma,n=3.,side='high'):
    if mean==-99:
        mean=np.mean(data)
    if sigma==-99:
        sigma=np.std(data)
    #does sigma clipping step
    if side=='high':
        ix=(data<(mean+n*sigma))
    elif side=='low':
        ix=(data>(mean-n*sigma))
    elif side=='both':
        ix=((data>(mean-n*sigma)) & (data<(mean+n*sigma)))
    return ix

def doMethod(method,idxr,idxb):
    if method=='both':
        cond=idxr*idxb
    elif method=='high':
        cond=idxr
    elif method=='low':
        cond=idxb
    return cond

def gmmRuntimeError():
    return -99,-99,-99,-99,-99,-99,False

def rsFit(x1,y,param,data_weights=None):
    # weights = 1/err
    mur,mub,sigr,sigb,alphar,alphab, conv = param
    # Take 3 sigma
    idx = DoClip(y,mur,sigr,n=3.,side='both')
    xfit, yfit, weight = x1[idx], y[idx], data_weights[idx]
    if len(xfit)<10:
        xfit, yfit, weight = x1, y, data_weights
    
    # initialize fitters
    g_init = models.Linear1D(1)
    fit = fitting.LevMarLSQFitter()
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=2.)
    with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        filtered_data, model = or_fit(g_init, xfit, yfit,weights=weight) # Fit
        # filtered_data, model = or_fit(g_init, x1, y)
    a,b = model._parameters
    return a,b

def doBackSubtraction(col,pdist,col_bg,p_bg,scale=1,minColor=-1,maxColor=4.1):
    """
    Purpose: Subtract the background on the colminColoror distrution weighted
    by the member probability(Pm=Pradial*Predshift)
    input:color array,member array, background histogram
    returns: data points and weights to be fed to GMM for fitting
    
    Obs: scale = pi*Rlambda**2/(pi.(r_out**2 - r_in**2))
    The Background Scale is just the area of the cluter at Rlambda divide by the area of the annuli. 
    These scale factor does not take into account the radial probability.
    """
    weight_hist, center, width = doBinColor(col,weight=pdist,x0=minColor,xend=maxColor,width=0.01,density=True)
    bg_hist, center, _ = doBinColor(col_bg,weight=p_bg,x0=minColor,xend=maxColor,width=width,density=True)

    scale_bg = len(col_bg)*width*scale ## Number of Background Galaxies
    scale_w = len(col)*width           ## Number of field+member Galaxies
    ##Obs.: scale=area correction = pi.Rlambda**2/(pi*(r_out**2-r_in**2)
    
    mu_bg, sig_bg = weighted_avg_and_std(col_bg, p_bg)
    paramBG = [mu_bg,sig_bg,scale_bg]
    # mean_bg = np.mean(scale_bg*bg_hist)
    # bg_interp = interp1d(center,scale_bg*bg_hist,fill_value=(mu_bg,mu_bg),bounds_error=False)

    bgsub_weighted=(scale_w*weight_hist-scale_bg*bg_hist)    ## member galaxies = Ngals*(PDF_weighted) - Nbg_gals*(Area_scale_factor)*(PDF_bg)
    
    ## Set negative values to zero
    bgsub_weighted[np.where(bgsub_weighted<0)] = 0

    ## Normalization factor
    kappa=np.sum(bgsub_weighted*width) ## in case of density=True, i.e. the fraction of subtracted galaxies

    ## make an array to feed gmm
    bgsub_weighted=np.array(bgsub_weighted)
    bgsub_weighted.shape=(len(weight_hist),1) 
    center.shape=(len(center),1)
    
    return center,bgsub_weighted,paramBG,kappa
    # return center,bgsub_weighted,bg_interp,kappa

def gmmLabel(mean):
    # mean is an array with length 2
    #choose which gaussian fit corresponds to red/blue/background
    if mean[0] != mean[1]:
        blue, red = np.argmin(mean), np.argmax(mean)
    else:
        blue, red = 0, 1
    return blue, red

def gmmValues(gmm):
    mu, sigma, alpha, conv =gmm.means_, np.sqrt(gmm.covars_), gmm.weights_, gmm.converged_
    
    blue, red = gmmLabel(mu)
    mur, sigr, alphar = float(mu[red]), float(sigma[red]), float(alpha[red])
    mub, sigb, alphab = float(mu[blue]), float(sigma[blue]), float(alpha[blue])

    return mur,mub,sigr,sigb,alphar,alphab,conv

def gmmHandleErrorFit(x,param,weights=None,mean=0.,nsigma=2,method='high',n_iter=3):
    print("Warning: GMM do not fit properly")
    mur,mub,sigr,sigb,alphar,alphab,conv = param
    col_fit, weights_fit,param_fit = x,weights,param
    
    
    ## in the case that assigns RS to Blue Cloud
    if (np.abs(mub-mean) < np.abs(mur-mean)):
        print("Blue Cloud was assigned as RS")
        param_fit = param[1],param[0],param[3],param[2],param[5],param[4],param[6]
        mur,mub,sigr,sigb,alphar,alphab,conv = param_fit

     ## In the case that does not converge: we trow out 3*sigma on both sides. Where sigma=0, mean=np.mean(color)
    if np.abs(mur-mean)>0.2 or (mur==-99) or (sigr > 0.5) or (np.abs(mur-mub)<0.01):
        print("Offset color:", (mur-mean))
        param = gmmRuntimeError()
        col_fit, weights_fit,param_fit = gmmSigmaClip_doFit(x,param_fit,weights=weights,nsigma=nsigma,method=method)

   ## in the case that the Blue Cloud vanishes
    if (alphab<0.05):
        print("Blue Cloud was vanished")
        param = gmmRuntimeError()
        col_fit, weights_fit,param_fit = gmmSigmaClip_doFit(x,param,weights=weights,nsigma=nsigma,method='high')
    
    ## in the case that assigns RS to Blue Cloud
    if (np.abs(mub-mean) < np.abs(mur-mean)):
        print("Blue Cloud was assigned as RS")
        param_fit = param[1],param[0],param[3],param[2],param[5],param[4],param[6]
        mur,mub,sigr,sigb,alphar,alphab,conv = param_fit

    ## If the conditions above does not work. Try a sigma clip
    mur,mub,sigr,sigb,alphar,alphab,conv = param_fit
    if np.abs(mur-mean)>0.2 or (sigr>0.5) or (alphab/alphar > 1.5) or (np.abs(mur-mub)<0.01) or (conv==False):
        param_fit = gmmRuntimeError()
        print("Error: It was not possilbe to handle the fit error")
    return col_fit, weights_fit, param_fit

def gmm_doFit(x,weights=None):
    gmm=mixture.GMM(n_components=2,tol=1e-7,n_iter=500)
    try:
        fit = gmm.fit(x, data_weights=weights)
        # fit = gmm.fit(x)
        mur,mub,sigr,sigb,alphar,alphab,conv = gmmValues(fit)
    except:
        mur,mub,sigr,sigb,alphar,alphab,conv = gmmRuntimeError()
    # print("convergence",conv)
    return mur,mub,sigr,sigb,alphar,alphab,conv    

def gmmClip_doFit(x,param,weights=None,nsigma=3,method='both'):
    # First Fit
    mur,mub,sigr,sigb,alphar,alphab,conv = param
    Flag = False    ## Error Flag
    # Clip
    # It does: color < mean_r + 3*sigma_r or/and color > mean_b - 3*sigma_r
    idxr, idxb = DoClip(x,mur,sigr,n=nsigma,side='high'), DoClip(x,mub,sigb,n=nsigma,side='low')
    cond = doMethod(method,idxr,idxb)       # return indices
    NonZeroValues = np.count_nonzero(cond)
    if NonZeroValues > 3:
        x1 = x[cond]
        x1.shape=(len(x1),1)
        if weights is not None:
            weights = weights[cond]
            weights.shape=(len(x1),1)
        try:
            # Second Fit
            gmm=mixture.GMM(n_components=2,tol=0.0000001,n_iter=500)
            fit1 = gmm.fit(x1, data_weights=weights)
            mur,mub,sigr,sigb,alphar,alphab,conv = gmmValues(fit1)
        except RuntimeError:
            Flag = True    
    else:
        Flag = True
    # By any chance it happens an error, it returns -99 values
    if Flag:
        mur,mub,sigr,sigb,alphar,alphab,conv = gmmRuntimeError()
        x1, weights=x,weights
    return x1,weights,[mur,mub,sigr,sigb,alphar,alphab,conv]

def gmmSigmaClip_doFit(x,param,weights=None,mean=0.,nsigma=3,method='both',n_iter=3):  
    """
    Purpose: Feed gmmMain_doFit; it does 3 fit iterantions with sigma cliping techinique
    method: both, high,low.Ex.: Both(3*sigmab < mub , mur < 3*sigmar)
    Obs.: If sigmar is below 0.2 it does not complete the n_iter.
    """  
    col_iter, weights_iter, param0 = x, weights, param
    count = 0
    ## It does sigma clip for n_iter. However, if sigma_r is below 0.2 it stops before n_iter.
    while count <= n_iter:
        # print("n_iter:",count)
        col0, w0, param0 = col_iter, weights_iter, param
        col_iter, weights_iter,param = gmmClip_doFit(col0,param0,weights=w0,nsigma=nsigma,method=method)
        count += 1
        if param[2]<0.2:
            break
        # print(count," run  sigma_r:",param[0])
    return col_iter,weights_iter,param

def gmmMain_doFit(col,weights=None,mean=0):
    """
    Purpose: Fit the red sequence and the blue cloud
    Method: Gaussian Mixture Modeling + sigma clipping
    """
    param = gmm_doFit(col,weights=weights) ## Initialization
    # 2 Sigma:
    col3,weights3,param = gmmSigmaClip_doFit(col,param,weights=weights,mean=mean,nsigma=2,method='both',n_iter=3)
    
    ## Check Fit
    mur,mub,sigr,sigb,alphar,alphab,conv = param
    NotWork = (np.abs(mur-mean)>0.2 or (mur==-99) or (sigr>0.5) or (np.abs(mub-mean) < np.abs(mur-mean)) or (alphab/alphar > 1) or (np.abs(mur-mub)<0.01))
    # Handling Errors
    if NotWork:
        col, weights,param = gmmHandleErrorFit(col,param,weights=weights,mean=mean,nsigma=2,method='both',n_iter=3)
    
    return [col3,weights3,param]

def gmmColorProb(color,kappa,param,paramBG,minColor=-1,maxColor=4.1):
    mur,mub,sigr,sigb,alphar,alphab,conv = param
    mu_bg,sig_bg,scale_bg = paramBG
    if param[0]!=-99:
        goodvals=np.where((color>minColor)&(color<maxColor))
        color_fit = color[goodvals]
        L_blue= PDF_color(color_fit,mub,sigb)
        L_red = PDF_color(color_fit,mur,sigr)
        Nbg = scale_bg*PDF_color(color_fit,mu_bg,sig_bg)
        
        # Nbg = bg_interp(color_fit)  ## Number of background galaxies in the bin 0.01
        # bg_interp = interp1d(center,bg_hist)      # Interpolation of the histogram
        ## Obs.: Is it better to fit the background color distribution?

        #calculate red/blue probabilities (see overleaf section 3.4)
        p_red_numerator=(alphar*L_red)*kappa  # Number of galaxies in the bin color width 0.01 
        p_red_denominator=((alphar*L_red))*kappa + Nbg
        p_red=p_red_numerator/p_red_denominator
        
        p_blue_numerator=(alphab*L_blue)*kappa
        # p_blue_denominator=((alphab*L_blue))*kappa + bg_interp(color_fit)
        p_blue_denominator=((alphab*L_blue))*kappa + Nbg
        p_blue=p_blue_numerator/p_blue_denominator

        # print "kappa, Nbkg:", kappa, bg_interp(color_fit).sum()

        #calculate color probabilities for each galaxy (see overleaf section 3.4)
        p_color_numerator=((alphab*L_blue)+(alphar*L_red))*kappa
        p_color_denominator=((alphab*L_blue)+(alphar*L_red))*kappa + Nbg
        p_color=p_color_numerator/p_color_denominator

        # Is_bgHist_None = (np.count_nonzero(bg_hist)<1)
        # if Is_bgHist_None:
        #     p_red=p_red_numerator/kappa; p_blue=p_blue_numerator/kappa; p_color=p_color_numerator/kappa
        
        p_blue.shape=(len(p_blue),);p_red.shape=(len(p_red),)
        p_color.shape=(len(p_color),)

        #Save color probabilities, prob = -1 if galaxy was cut out by color-specific crazy color cuts
        
        tmp_Pred=(-1.)*np.ones_like(color)
        tmp_Pred[goodvals]=p_red
        tmp_Pblue=(-1.)*np.ones_like(color)
        tmp_Pblue[goodvals]=p_blue
        tmp_Pcolor=(-1.)*np.ones_like(color)
        tmp_Pcolor[goodvals]=p_color
    else:
        tmp_Pcolor, tmp_Pred, tmp_Pblue = (-99.)*np.ones_like(color),(-99.)*np.ones_like(color),(-99.)*np.ones_like(color)
    return tmp_Pcolor, tmp_Pred, tmp_Pblue


def fitFullRS(color,band2,bkg,Pdist,cluster_ID,cluster_Z,galaxyID,hostID,r_cl,maglim,param_rs=None,scale_vec=None,Pmem_lim=0.0,Nlabel=0,LabelSave='',minColor=-1,maxColor=4.1,tol=0.0000001,plot=0):
    #Full GMM calculation for single observed color
    #Pdist=P_radial*P_redshift
    hostID_bg,color_bg,mag_bg,pz_bg = bkg 
    ## Initializing variables
    slope=[]; yint=[]; RS_Color=[]; mu_r=[]; mu_b=[]
    sigma_r=[]; sigma_b=[]; alpha_r=[]; alpha_b=[]
    Pred=[]; Pblue=[]; Pcolor=[]; converged=[]; probgalid = []
    
    cluster_Z=np.array(cluster_Z)

    ## Define Color Labels
    ## Legend: color_gr (Nlabel=0), color_ri (Nlabel=1), color_iz(Nlabel=2)
    labelx, labely, labely_offset, saveName = colorLabel(Nlabel)
    saveName_offset = 'offset_%s'%(saveName.split('color_')[1])
    for i in range(len(cluster_ID)):
        x=cluster_ID[i]
        name="RM-{}".format(x)
        print 'Cluster %s'%(name)
        name=LabelSave+name
        idx_cls = np.where(cluster_ID==x)
        zcl=cluster_Z[idx_cls]            ## Cluster Redshift
        rcl=r_cl[idx_cls]                 ## Cluster Radius
        mag_lim = float(maglim[idx_cls])  ## Mag lim
        expred,paramRS = auxRS(param_rs,idx_cls) 

        ## Member Galaxies: Cut(Pmemb >= lim1, Color < max color, mag_lim)
        idx1, = np.where((hostID==x) & (Pdist>=Pmem_lim) & (color<maxColor))
        glxid1,color1,mag1,distprob1 =galaxyID[idx1],color[idx1],band2[idx1],Pdist[idx1]

        # Magnitude Cut
        # print("magnitude cut:",mag_lim)
        idx2, = np.where(mag1<=mag_lim)
        glxid,color2,mag2,distprob2=glxid1[idx2],color1[idx2],mag1[idx2],distprob1[idx2]
      
        ## Background Galaxies
        w,=np.where(hostID_bg==i)
        mag_bg1, color_bg1, pz_bg1 = mag_bg[w], color_bg[w], pz_bg[i]
        # cut
        w, = np.where(mag_bg1<mag_lim)
        mag_bg1, color_bg1, pz_bg1 = mag_bg[w], color_bg[w], pz_bg1[w]

        ## Define Color Offsets
        minOffset, maxOffset = (minColor-expred),(maxColor-expred)
        offset1 = doRS_Offset(mag1,color1,paramRS)
        offset2 = doRS_Offset(mag2,color2,paramRS)
        bg_offset = doRS_Offset(mag_bg1,color_bg1,paramRS)

        ## Define Background Histogram
        scale_bg = getScale(scale_vec,i)
        # color_bg1, bg_offset = np.zeros_like(color_bg1), np.zeros_like(bg_offset)
        # bg_hist,_,_ = doBinColor(color_bg1,weight=pz_bg1,x0=minColor,xend=maxColor,width=0.01)
        # bg_hist_offset,_,_ = doBinColor(bg_offset,weight=pz_bg1,x0=minOffset,xend=maxOffset,width=0.01)
        
        if len(glxid) >= 3:
            # Background Subtraction & GMM + Sigma Clip Fit
            col_sub,colweights,paramBG,kappa = doBackSubtraction(color2,distprob2,color_bg1,pz_bg1,scale=scale_bg,minColor=minColor,maxColor=maxColor)
            col_sub0,colweights0,paramBG_Offset,kappa0 = doBackSubtraction(offset2,distprob2,bg_offset,pz_bg1,scale=scale_bg,minColor=minOffset,maxColor=maxOffset)
            # col_sub2,colweights2,color_sub_distrib_interp2,kappa = doBackSubtraction(offset1,None,bg_hist)
            
            ## Do GMM FIT (Sigma Clip)
            col3,weights3,param = gmmMain_doFit(col_sub,weights=colweights,mean=expred)
            col30,weights30,param0 = gmmMain_doFit(col_sub0,weights=colweights0,mean=0)
            
            ## mean and sig are default values to handle the fit's error
            ## Color Probabilities
            tmp_Pcolor, tmp_Pred, tmp_Pblue = gmmColorProb(color1,kappa,param,paramBG,minColor=minColor,maxColor=maxColor)
            tmp_Pcolor0, tmp_Pred0, tmp_Pblue0 = gmmColorProb(offset1,kappa0,param0,paramBG_Offset,minColor=minOffset,maxColor=maxOffset)

            ## RS Fit
            paramRS_cls = rsFit(mag2,color2,param,data_weights=tmp_Pred[idx2]*distprob2)
            rs_color = doRS_Color(mag1,paramRS_cls)
            
            ##    Testing without weights
            # col3_2,weights3_2,param2 = gmmMain_doFit(col_sub2,weights=colweights2,mean=expred) 
            # make2ColorHist(offset1,offset1,zcl,data_weights=distprob2,param=param,param2=param2,labelx=labely,description=name,hpc=1)
            ## Plots
            if plot:
                # Color Distribution
                # if tmp_Pred[-1]!=-99:
                #     doPlot_colorModel(color2,tmp_Pred[idx2],tmp_Pblue[idx2],tmp_Pcolor[idx2],distprob2,param,kappa,zcl,mag_lim,labelx=labely,name=name,save=saveName,hpc=1)
                #     # doPlot_colorModel(offset2,tmp_Pred0[idx2],tmp_Pblue0[idx2],tmp_Pcolor0[idx2],distprob2,param0,kappa0,zcl,mag_lim,labelx=labely_offset,name=name,save=saveName_offset,hpc=1)
                # doPlot_colorModel_bg(color2,color_bg1,tmp_Pred[idx2],tmp_Pblue[idx2],tmp_Pcolor[idx2],distprob2,scale_bg,pz_bg1,param,kappa,zcl,mag_lim,expred,labelx=labely,name=name,save=saveName,hpc=1)
                doPlot_colorModel_bg(offset2,bg_offset,tmp_Pred0[idx2],tmp_Pblue0[idx2],tmp_Pcolor0[idx2],distprob2,scale_bg,pz_bg1,param0,kappa0,zcl,mag_lim,0,labelx=labely_offset,name=name,save=saveName_offset,hpc=1)
                # RS_plot_fit(color1,mag1,param,paramRS_cls,tmp_Pred,tmp_Pblue,prob_weights=distprob1,maglim=mag_lim,labelx=labelx,labely=labely,Name=name,save=saveName,zcls=zcl,hpc=1)
                RS_plot_fit(offset1,mag1,param0,[0,0],tmp_Pred,tmp_Pblue,prob_weights=distprob1,maglim=mag_lim,labelx=labelx,labely=labely_offset,Name=name,zcls=zcl,save=saveName_offset,hpc=1)

            # Save
            mur,mub,sigr,sigb,alphar,alphab,conv = param
            mur0,mub0,sigr0,sigb0,alphar0,alphab0,conv0 = param0
        
        else:
            mur,mub,sigr,sigb,alphar,alphab,conv = gmmRuntimeError()
            mur0,mub0,sigr0,sigb0,alphar0,alphab0,conv0 = gmmRuntimeError()
            tmp_Pblue, tmp_Pred, tmp_Pcolor = -99*np.ones_like(glxid1),-99*np.ones_like(glxid1),-99*np.ones_like(glxid1) 
    
        # mu_r.append(mur); sigma_r.append(sigr); alpha_r.append(alphar)
        # mu_b.append(mub); sigma_b.append(sigb); alpha_b.append(alphab)

        mu_r.append(mur); sigma_r.append(sigr0); alpha_r.append(alphar0)
        mu_b.append(mub); sigma_b.append(sigb0); alpha_b.append(alphab0)
        
        Pred.append(tmp_Pred);     Pblue.append(tmp_Pblue)
        Pcolor.append(tmp_Pcolor); probgalid.append(glxid1)
        converged.append(conv)

        slope.append(paramRS_cls[0]); yint.append(paramRS_cls[1]); RS_Color.append(rs_color)
    ## Replace these lines on the future
    mu_r=np.array((mu_r))
    mu_b=np.array((mu_b))
    sigma_r=np.array((sigma_r))
    sigma_b=np.array((sigma_b))
    alpha_r=np.array((alpha_r))
    alpha_b=np.array((alpha_b))
    Pred = doSublistToArray(np.array(Pred))
    Pblue= doSublistToArray(np.array(Pblue))
    probgalid=doSublistToArray(np.array(probgalid))
    Pcolor=doSublistToArray(np.array(Pcolor))
    converged=np.array((converged))
    slope=np.array((slope))
    yint=np.array((yint))
    RS_Color = doSublistToArray(np.array(RS_Color))
    return mu_r,mu_b,sigma_r,sigma_b,alpha_r,alpha_b,converged,Pred,Pblue,Pcolor,probgalid,slope,yint,RS_Color

    #,colfit,colweights,n_subtracted#,colfit2,colweights2,colfit3,colweights3


# Make the color bar and legend size
# import seaborn as sns
# import matplotlib.pyplot as plt
# iris = sns.load_dataset("iris")

# plt.scatter(iris.sepal_width, iris.sepal_length, 
#             c = iris.petal_length, s=(iris.petal_width**2)*60, cmap="viridis")
# ax = plt.gca()

# plt.colorbar(label="petal_length")
# plt.xlabel("sepal_width")
# plt.ylabel("sepal_length")

# #make a legend:
# pws = [0.5, 1, 1.5, 2., 2.5]
# for pw in pws:
#     plt.scatter([], [], s=(pw**2)*60, c="k",label=str(pw))

# h, l = plt.gca().get_legend_handles_labels()
# plt.legend(h[1:], l[1:], labelspacing=1.2, title="petal_width", borderpad=1, 
#             frameon=True, framealpha=0.6, edgecolor="k", facecolor="w")

# plt.show()

# # Trash

# def RS_hist(col,pdist):
#     weight_hist, center, width = doBinColor(col,pdist)

#     ## make an array to feed gmm
#     weight_hist.shape=(len(weight_hist),1) 
#     center_fit.shape=(len(center),1)

#     return center_fit,bgsub_weighted

# def IsValidCondition(ix):
#     # Is it non zero length?
#     sub_size = np.size(ix) - np.count_nonzero(ix)
#     return sub_size != 0

# def local_bg_histograms(bkg_vec,width=0.01):
#     ang_diam_dist,host_id,gmag,rmag,imag,zmag,pz = bkg_vec
#     hist_bins=np.arange(-1,4.1,width)
#     m1uniq=np.arange(ang_diam_dist.size).astype(int)
    
#     gr_hists=[];ri_hists=[];iz_hists=[]
#     for x in m1uniq:
#         w=np.where(host_id==m1uniq[x])
#         gmag1=gmag[w]; rmag1=rmag[w]
#         imag1=imag[w]; zmag1=zmag[w]
#         pz1=pz[x]
#         gr=gmag1-rmag1; ri=rmag1-imag1
#         iz=imag1-zmag1
#         gr_h,gr_e=np.histogram(gr,bins=hist_bins,weights=pz1,density=True)
#         ri_h,ri_e=np.histogram(ri,bins=hist_bins,weights=pz1,density=True)
#         iz_h,iz_e=np.histogram(iz,bins=hist_bins,weights=pz1,density=True)
#         gr_hists.append(gr_h)
#         ri_hists.append(ri_h)
#         iz_hists.append(iz_h)
#     return np.array(gr_hists), np.array(ri_hists), np.array(iz_hists)


# def make2ColorHist(X,X2,z_cl,data_weights=None,param=None,param2=None,labelx='$r-i$',description=None,hpc=0):
#     if hpc:
#         savedir='/data/des61.a/data/johnny/clusters/test/plots/'#/redmapper_y1a1/localsub/test'
#     else:
#         savedir='./plots/'
#     mur,mub,sigr,sigb,alphar,alphab,conv = param
#     a,b = (mub-3*sigb-0.3), (mur+3*sigr+0.3)
#     y = np.linspace(a,b,1000)

#     label1, label2 =  r'$P_{mem} \times N_{gal}$',r'$N_{gal}$'
#     plt.title("Color Distribution: "+description)
#     out= description+'_fit.png'

#     fithist, center, width = doBinColor(X,weight=data_weights,x0=a,xend=b,width=0.075)
#     fithist2, center2, width2 = doBinColor(X2,x0=a,xend=b,width=0.075)
    
#     norm = np.sum(fithist*width)
#     norm2 = np.sum(fithist2*width2)
    
#     blue, red = 'steelblue', 'lightcoral'
#     blue2, red2 = '#024794','#AF2B2F'
#     plt.bar(center2,fithist2,align='center',width=width2,edgecolor='white',linewidth=1.2,facecolor=red,label=label2,alpha=0.7)
#     if param2 is not None:
#         mur2,mub2,sigr2,sigb2,alphar2,alphab2, conv2 = param2
#         plt.plot(y,norm2*stats.norm.pdf(y,mur2,sigr2)*alphar2,color=red2)
#         plt.plot(y,norm2*stats.norm.pdf(y,mub2,sigb2)*alphab2,color=blue2)

#     plt.bar(center,fithist,align='center',width=width,edgecolor='white',linewidth=1.2,facecolor=blue, label=label1, alpha=0.9)
#     plt.plot(y,norm*stats.norm.pdf(y,mur,sigr)*alphar,color=red2,linestyle="--")
#     plt.plot(y,norm*stats.norm.pdf(y,mub,sigb)*alphab,color=blue2,linestyle="--")
#     plt.figtext(0.3,0.85,'$\\sigma_r,1=%.2f \quad \\sigma_r,2=%.2f$ \n $\\sigma_b,1=%.2f \quad \\sigma_b,1=%.2f$ \n $z=%.2f \quad mag_{lim} = 21$'%(sigr,sigr2,sigb,sigb2,z_cl),ha='right', va='top')
#     plt.legend(loc='upper right')
#     ymin,ymax=plt.ylim()
#     plt.xlabel(labelx)
    
#     out_name = savedir+(description)+'_color_%s.png'%(labelx.split('$')[1])
#     plt.savefig(out_name)
#     plt.close()

# ### This is the function that call the others.
# def fitFull(color,band2,background_histograms,Pdist,cluster_ID,cluster_Z,galaxyID,hostID,r_cl,maglim,rs_mean=None,Pmem_lim=0.0,color_label=('$r$','$r-i$'),minColor=-1,maxColor=4.1,tol=0.0000001):
#     #Full GMM calculation for single observed color
#     #for g-r, band1=galmagG, band2=galmagR
#     #Pdist=P_radial*P_redshift
    
#     ## Initializing variables
#     slope=[]; yint=[]; RS_Color=[]; mu_r=[]; mu_b=[]
#     sigma_r=[]; sigma_b=[]; alpha_r=[]; alpha_b=[]
#     Pred=[]; Pblue=[]; Pcolor=[]; converged=[]; probgalid = []
    
#     cluster_Z=np.array(cluster_Z)
#     for x in cluster_ID:
#         name="RM-{}".format(x)
#         idx_cls = np.where(cluster_ID==x)
#         zcl=cluster_Z[idx_cls]            ## Cluster Redshift
#         rcl=r_cl[idx_cls]                 ## Cluster Radius
#         mag_lim = float(maglim[idx_cls])         ## Mag lim
#         expred = IsRSmean(rs_mean,idx_cls)## Expected RS mean

#         ## Member Galaxies, Pmemb >= lim1
#         idx0 = np.where((hostID==x)&(Pdist>=Pmem_lim))
#         color0,glxid0,magr0,distprob0=color[idx0],galaxyID[idx0],band2[idx0],Pdist[idx0]
        
#         ## Member Galaxies, Pmemb >= lim1, Color < max color
#         idx1 = np.where((hostID==x)&(Pdist>=Pmem_lim)&(color<maxColor))
#         glxid1,color1,magr1,distprob1 =galaxyID[idx1],color[idx1],band2[idx1],Pdist[idx1]

#         ## Magnitude Cut
#         print("magnitude cut:",mag_lim)
#         idx2 = np.where(magr1<=mag_lim)
#         glxid,color2,magr2,distprob2=glxid1[idx2],color1[idx2],magr1[idx2],distprob1[idx2]
        
#         ## Define Background Histogram
#         if background_histograms is None: #JCB added .any() w/o throws error from astropy fits
#             hist_bins=np.arange(minColor,maxColor,0.01)
#             bg_hist=np.zeros_like(hist_bins)
#             bg_hist=bg_hist[:-1]
#             bg_hist=np.array([bg_hist])
#         else:
#             bg_hist=background_histograms[np.where(cluster_ID==x)]
#             bg_hist=bg_hist[0]

#         if len(glxid) >= 3:
#             # Background Subtraction & GMM + Sigma Clip Fit
#             col_sub,colweights,color_sub_distrib_interp,kappa = doBackSubtraction(color2,distprob2,bg_hist,minColor=minColor,maxColor=maxColor)
#             # col_sub2,colweights2,color_sub_distrib_interp2,kappa = doBackSubtraction(color2,None,bg_hist)
            
#             ## Do GMM FIT (Sigma Clip)
#             col3,weights3,param = gmmMain_doFit(col_sub,weights=colweights,mean=expred)
            
#             ## mean and sig are default values to handle the fit's error
#             ## Color Probabilities
#             tmp_Pcolor, tmp_Pred, tmp_Pblue = gmmColorProb(color1,bg_hist,kappa,param,minColor=minColor,maxColor=maxColor)
            

#             # Color Distribution
#             labelx, labely = color_label 
#             # doHist_Color(color2,param,tmp_Pred[idx2],tmp_Pblue[idx2],tmp_Pcolor[idx2],kappa,zcl,weights1=distprob2,FLAG=param[-1],labelx=labely,description=name,hpc=1)
#             # doPlot_colorModel(color2,tmp_Pred[idx2],tmp_Pblue[idx2],tmp_Pcolor[idx2],distprob2,param,kappa,zcl,mag_lim,labelx=labely,name=name,hpc=1)
            
#             ## RS Fit
#             paramRS = rsFit(magr2,color2,param,data_weights=tmp_Pred[idx2]*distprob2)
#             rs_color = doRS_Color(magr0,paramRS)
#             RS_plot_fit(color1,magr1,param,paramRS,tmp_Pred,tmp_Pblue,prob_weights=distprob1,maglim=mag_lim,labelx=labelx,labely=labely,Name=name,zcls=zcl,hpc=1)
            
#             ##    Testing without weights
#             # col3_2,weights3_2,param2 = gmmMain_doFit(col_sub2,weights=colweights2,mean=expred) 
#             # make2ColorHist(color2,color2,zcl,data_weights=distprob2,param=param,param2=param2,labelx=labely,description=name,hpc=1)

#             # Save
#             mur,mub,sigr,sigb,alphar,alphab,conv = param
            
#             mu_r.append(mur); sigma_r.append(sigr); alpha_r.append(alphar)
#             mu_b.append(mub); sigma_b.append(sigb); alpha_b.append(alphab)
            
#             Pred.append(tmp_Pred);     Pblue.append(tmp_Pblue)
#             Pcolor.append(tmp_Pcolor); probgalid.append(glxid0)
#             converged.append(conv)

#             slope.append(paramRS[0]); yint.append(paramRS[1]); RS_Color.append(rs_color)

#     ## Replace these lines on the future
#     mu_r=np.array((mu_r))
#     mu_b=np.array((mu_b))
#     sigma_r=np.array((sigma_r))
#     sigma_b=np.array((sigma_b))
#     alpha_r=np.array((alpha_r))
#     alpha_b=np.array((alpha_b))
#     Pred = doSublistToArray(np.array(Pred))
#     Pblue= doSublistToArray(np.array(Pblue))
#     probgalid=doSublistToArray(np.array(probgalid))
#     Pcolor=doSublistToArray(np.array(Pcolor))
#     converged=np.array((converged))
#     slope=np.array((slope))
#     yint=np.array((yint))
#     RS_Color = doSublistToArray(np.array(RS_Color))
#     return mu_r,mu_b,sigma_r,sigma_b,alpha_r,alpha_b,converged,Pred,Pblue,Pcolor,probgalid,slope,yint,RS_Color

# def gmmColorProb(color,bg_interp,kappa,param,minColor=-1,maxColor=4.1):
#     mur,mub,sigr,sigb,alphar,alphab,conv = param
#     if param[0]!=-99:
#         goodvals=np.where((color>minColor)&(color<maxColor))
#         color_fit = color[goodvals]
#         L_blue= PDF_color(color_fit,mub,sigb)
#         L_red = PDF_color(color_fit,mur,sigr)
        
#         # bg_interp = interp1d(center,bg_hist)      # Interpolation of the histogram
#         ## Obs.: Is it better to fit the background color distribution?
#         #calculate red/blue probabilities (see overleaf section 3.4)
        
#         Nbg = bg_interp(color_fit)  ## Number of background galaxies in the bin 0.01

#         p_red_numerator=(alphar*L_red)*kappa  # Number of galaxies in the bin 0.01
#         p_red_denominator=((alphar*L_red))*kappa + Nbg
#         p_red=p_red_numerator/p_red_denominator
        
#         p_blue_numerator=(alphab*L_blue)*kappa
#         # p_blue_denominator=((alphab*L_blue))*kappa + bg_interp(color_fit)
#         p_blue_denominator=((alphab*L_blue))*kappa + Nbg
#         p_blue=p_blue_numerator/p_blue_denominator

#         # print "kappa, Nbkg:", kappa, bg_interp(color_fit).sum()

#         #calculate color probabilities for each galaxy (see overleaf section 3.4)
#         p_color_numerator=((alphab*L_blue)+(alphar*L_red))*kappa
#         p_color_denominator=((alphab*L_blue)+(alphar*L_red))*kappa + Nbg
#         p_color=p_color_numerator/p_color_denominator

#         # Is_bgHist_None = (np.count_nonzero(bg_hist)<1)
#         # if Is_bgHist_None:
#         #     p_red=p_red_numerator/kappa; p_blue=p_blue_numerator/kappa; p_color=p_color_numerator/kappa
        
#         p_blue.shape=(len(p_blue),);p_red.shape=(len(p_red),)
#         p_color.shape=(len(p_color),)

#         #Save color probabilities, prob = -1 if galaxy was cut out by color-specific crazy color cuts
        
#         tmp_Pred=(-1.)*np.ones_like(color)
#         tmp_Pred[goodvals]=p_red
#         tmp_Pblue=(-1.)*np.ones_like(color)
#         tmp_Pblue[goodvals]=p_blue
#         tmp_Pcolor=(-1.)*np.ones_like(color)
#         tmp_Pcolor[goodvals]=p_color
#     else:
#         tmp_Pcolor, tmp_Pred, tmp_Pblue = (-99.)*np.ones_like(color),(-99.)*np.ones_like(color),(-99.)*np.ones_like(color)
#     return tmp_Pcolor, tmp_Pred, tmp_Pblue

