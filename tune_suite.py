#!/usr/bin/env python
from __future__ import print_function
import numpy as np

# suite of functions that can do different tune calculations

# calculate the simple x y z tunes just from fft
def basic_tunes(coords):
    # coords has shape (6,n)
    n = coords.shape[1]
    df = 1.0/n
    # get maximum amplitude location
    maxn = n//2
    xt = np.abs(np.fft.fft(coords[0,:]))
    # probably don't want 0 frequency
    locmax = np.argmax(xt[0:maxn])
    xtune = (locmax)*df
    yt = np.abs(np.fft.fft(coords[2,:]))
    locmax = np.argmax(yt[0:maxn])
    ytune = (locmax)*df
    zt = np.abs(np.fft.fft(coords[4,:]))
    locmax = np.argmax(zt[0:maxn])
    ztune = (locmax)*df
    return (xtune,ytune,ztune)

# calculate tunes using the CFT algorithm
def cft_tunes(coords, search_range=((0.0,0.5),(0.0,0.5),(0.0,0.5))):
    # coords has shape (6,n)
    n = coords.shape[1]
    #if n>100:
    #    print "warning! you have a lot of points.  This will be slow"

    # normal FFT precision is 1/n, CFT gives addition 1/n factor
    df = 1.0/n**2
    t = np.arange(n,dtype='d')

    # loop over x,y,z
    tunes = np.zeros(3)
    for pln in range(3):
        f = np.arange(search_range[pln][0],search_range[pln][1],df)
        nf = len(f)
        cft = np.zeros(nf,dtype='d')
        idxcoord = 2*pln

        for i in range(nf):
            expfact = np.exp(-2.0*np.pi*(1.0j)*f[i]*t)
            cft[i] = abs(np.dot(expfact, coords[idxcoord,:]))

        cftmaxloc = np.argmax(cft)
        tunes[pln] = f[cftmaxloc]
    
    return tuple(tunes)

# get interpolated tunes
# filter is a list of tunes which are excluded from selection as the maximum tune
def interp_tunes(coords, search_range=((0.0,0.5),(0.0,0.5),(0.0,0.5)), filter=[[],[],[]]):
    #print('interp_tunes: coords.shape: ', coords.shape)
    # coords has shape (6,n)
    n = coords.shape[1]
    #print('interp_tunes: n: ', n)
    if n <= 0:
        raise RuntimeError("WTF!  n<=0!!!, n:{}".format(n))
    # loop pver x. y, z
    tunes = np.zeros(3)
    maxn = n//2
    #print('interp_tunes: ', maxn)
    df = 1.0/n
    #print('interp_tunes: df: ', df)
    f = np.arange(n,dtype='d')/n
    for pln in range(0,6,2):
        # filter frequencies for this plane
        plnfilt = filter[pln//2]
        xt = abs(np.fft.fft(coords[pln, :]))
        xtprime = xt[1:] - xt[0:-1]
        # loop over frequencies to filter
        for ff in filter[pln//2]:
            #print('cutting frequency ', ff)
            fbin = int(ff/df)
            #print('fbin: ', fbin)
            i0 = fbin-2
            if i0 < 0:
                i0 = 0
            i1 = fbin+3
            if i1 >= maxn:
                i1 = maxn
            for i in range(i0, i1):
                #print(xt[i], end='')
                if i == fbin:
                    print('*', end='')
                print(' ', end='')
            #print()
            
            # check lower side to see if this is a hump
            #print('checking bins ', fbin, ' and lower')
            if fbin == 0:
                xt[fbin] = 0.0
            elif fbin > 0 and xtprime[fbin-1] > 0:
                # yes, zero out the left side until derivative becomes negative
                hbin = fbin
                #print('hbin: ', hbin, ' xt[hbin]: ', xt[hbin], ' xtprime[hbin-1]: ', xtprime[hbin-1])
                while hbin > 0 and xtprime[hbin-1] > 0:
                    #print(' killing ', hbin)
                    xt[hbin] = 0.0
                    hbin = hbin - 1
            # check upper side
            if fbin == maxn-1:
                xt[fbin] = 0.0
            elif fbin < maxn-1 and xtprime[fbin] < 0:
                #print('checking bins ', fbin, ' and higher')
                # zero out right side until derivative becomes positive
                hbin = fbin
                while hbin < maxn-1 and xtprime[hbin] < 0:
                    #print('killing ', hbin)
                    xt[hbin+1] = 0.0
                    hbin = hbin + 1

        locmax = np.argmax(xt[0:maxn])
        #print('interp_tunes: locmax: ', locmax)
        tune = -999.0
        if locmax == 0:
            tune = 0.0
        else:
            f0 = xt[locmax]
            #print('interp_tunes: pln: ', pln, ', f0: ', f0)
            if xt[locmax-1] > xt[locmax+1]:
                dir=-1.0
                xtp2 = xt[locmax-1]
            else:
                dir=1.0
                xtp2 = xt[locmax+1]
            #print('interp_tunes: dir: ', dir, ', xtp2: ', xtp2)
            tune = df*locmax + df*dir*(xtp2/(xt[locmax]+xtp2))

        tunes[pln//2] = tune

    return tuple(tunes)

def interp_tunes_nonzero(coords, search_range=((0.0,0.5),(0.0,0.5),(0.0,0.5))):
    #print('interp_tunes: coords.shape: ', coords.shape)
    # coords has shape (6,n)
    n = coords.shape[1]
    #print('interp_tunes: n: ', n)
    if n <= 0:
        raise RuntimeError("WTF!  n<=0!!!, n:{}".format(n))
    # loop pver x. y, z
    tunes = np.zeros(3)
    maxn = n//2
    print('interp_tunes: ', maxn)
    df = 1.0/n
    #print('interp_tunes: df: ', df)
    f = np.arange(n,dtype='d')/n
    for pln in range(0,6,2):
        xt = abs(np.fft.fft(coords[pln, :]))
        # avoid 0
        locmax = np.argmax(xt[1:maxn])+1
        #print('interp_tunes: locmax: ', locmax)
        tune = -999.0
        f0 = xt[locmax]
        #print('interp_tunes: pln: ', pln, ', f0: ', f0)
        if xt[locmax-1] > xt[locmax+1]:
            dir=-1.0
            xtp2 = xt[locmax-1]
        else:
            dir=1.0
            xtp2 = xt[locmax+1]
            #print('interp_tunes: dir: ', dir, ', xtp2: ', xtp2)
            tune = df*locmax + df*dir*(xtp2/(xt[locmax]+xtp2))

        tunes[pln//2] = tune

    return tuple(tunes)
        

# get the (fractional) tunes of a set of coordinates from a single track
def refined_tunes(coords):
    # coords has shape (6,n)
    n = coords.shape[1]
    df = 1.0/n

    if n <= 100:
        # really, if there are less than 100 points, just use
        # the cft tunes
        ctunes = cft_tunes(coords)
        return ctunes

    # first get the basic tunes
    btunes = basic_tunes(coords)

    if np.any(btunes == 0.0):
        # if some of the basic tunes are 0.0, then there was
        # some problem and it's probably not right. Get the CFT tunes
        # for the first 100 points.
        ctunes = cft_tunes(coords[:,:100])
        xrange = (ctunes[0]-2.0*df, ctunes[0]+2.0*df)
        yrange = (ctunes[1]-2.0*df, ctunes[1]+2.0*df)
        zrange = (ctunes[2]-2.0*df, ctunes[2]+2.0*df)
    else:
        xrange = (btunes[0]-2.0*df, btunes[0]+2.0*df)
        yrange = (btunes[1]-2.0*df, btunes[1]+2.0*df)
        zrange = (btunes[2]-2.0*df, btunes[2]+2.0*df)
        
    tunerange = (xrange,yrange,zrange)
    ctunes = cft_tunes(coords, tunerange)
    return ctunes
