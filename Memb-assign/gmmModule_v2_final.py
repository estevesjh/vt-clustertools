#!/usr/bin/env python -W ignore::DeprecationWarning
# gmmModule.py
# Purpose: Fit a gaussian mixture modelling on the color distribution of a given galaxy cluster catalog. 
## v2 - Study of the background Subtraction

# Description: In order to do the fit. First, it subtracts the background color distribution. 
# Second, produce a color distribution subtracted weighted by the membership probability.
# Third, it fit two gaussians, the red sequence and the blue cloud. 
# Finnaly, it produces the outputs that are the color probabilities (Pred,Pblue and Pcolor) and the gaussian parameter (mean_i,sigma_i,amplitude_i).

# input: color_data,mag_data,background_histograms,Membership Probability,cluster_ID,cluster_Z,galaxyID,hostID,r_cl,maxcol,maglim
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

def IsRSmean(rs_mean,idx):
    if rs_mean is not None:
        expred=float(rs_mean[idx]) ## Mean RS Color
    else:
        expred=0
    return expred

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
        # param_fit = gmmRuntimeError()
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
        
        #calculate red/blue probabilities (see overleaf section 3.4)
        p_red_numerator=(alphar*L_red)*kappa  # Number of galaxies in the bin color width 0.01 
        p_red_denominator=((alphar*L_red))*kappa + Nbg
        p_red=p_red_numerator/p_red_denominator
        
        p_blue_numerator=(alphab*L_blue)*kappa
        p_blue_denominator=((alphab*L_blue))*kappa + Nbg
        p_blue=p_blue_numerator/p_blue_denominator

        #calculate color probabilities for each galaxy (see overleaf section 3.4)
        p_color_numerator=((alphab*L_blue)+(alphar*L_red))*kappa
        p_color_denominator=((alphab*L_blue)+(alphar*L_red))*kappa + Nbg
        p_color=p_color_numerator/p_color_denominator

        p_blue.shape=(len(p_blue),);p_red.shape=(len(p_red),)
        p_color.shape=(len(p_color),)

        #Save color probabilities, prob = -1 if galaxy was cut out by color-specific crazy color cuts
        
        tmp_Pred=(-1.)*np.ones_like(color);tmp_Pblue=(-1.)*np.ones_like(color);tmp_Pcolor=(-1.)*np.ones_like(color)
        tmp_Pred[goodvals]=p_red
        tmp_Pblue[goodvals]=p_blue
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
            
            ## Plots
            if plot:
                # Color Distribution
                # if tmp_Pred[-1]!=-99:
                #     doPlot_colorModel(color2,tmp_Pred[idx2],tmp_Pblue[idx2],tmp_Pcolor[idx2],distprob2,param,kappa,zcl,mag_lim,labelx=labely,name=name,save=saveName,hpc=1)
                #     # doPlot_colorModel(offset2,tmp_Pred0[idx2],tmp_Pblue0[idx2],tmp_Pcolor0[idx2],distprob2,param0,kappa0,zcl,mag_lim,labelx=labely_offset,name=name,save=saveName_offset,hpc=1)
                doPlot.colorModel_bg(color2,color_bg1,tmp_Pred[idx2],tmp_Pblue[idx2],tmp_Pcolor[idx2],distprob2,scale_bg,pz_bg1,param,kappa,zcl,mag_lim,expred,labelx=labely,name=name,save=saveName,hpc=1)
                # doPlot_colorModel_bg(offset2,bg_offset,tmp_Pred0[idx2],tmp_Pblue0[idx2],tmp_Pcolor0[idx2],distprob2,scale_bg,pz_bg1,param0,kappa0,zcl,mag_lim,0,labelx=labely_offset,name=name,save=saveName_offset,hpc=1)
                doPlot.RS_fit(color1,mag1,param,paramRS_cls,tmp_Pred,tmp_Pblue,prob_weights=distprob1,maglim=mag_lim,labelx=labelx,labely=labely,Name=name,save=saveName,zcls=zcl,hpc=1)
                # RS_plot_fit(offset1,mag1,param0,[0,0],tmp_Pred,tmp_Pblue,prob_weights=distprob1,maglim=mag_lim,labelx=labelx,labely=labely_offset,Name=name,zcls=zcl,save=saveName_offset,hpc=1)

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