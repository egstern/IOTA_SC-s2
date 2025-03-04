#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import re
import glob
import mpi4py
import mpi4py.MPI as MPI
import h5py
import numpy as np
from  tune_suite import *

DEBUG=1

##  set the length of the tune calculation (in turns) and the number
##  of turns between each calculation.

# tune_length = 128 this seemed to be a little small and resulted in artifacts
tune_length = 1024
tune_steps = 16



if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("usage: calc_tunes.py tracks-filename")

    os.environ["HDF5_DISABLE_VERSION_CHECK"] = "2"

    commsize = MPI.COMM_WORLD.Get_size()
    myrank = MPI.COMM_WORLD.Get_rank()

    tracks_file = sys.argv[1]
    if DEBUG and myrank == 0:
        print("Reading tracks from file ", tracks_file)

    tunes_file = os.path.splitext(tracks_file)[0]+"_tunes"
    h5f = h5py.File(tracks_file, 'r')

    # collect tunes in a dictionary indexed by particle ID containing
    # a tuple of ([starting turn], [xtunes], [ytunes], [ltunes])

    tracks = h5f.get("track_coords")
    # tracks is array <turns> x <particles> x <coords>
    print("track_coords.shape: ", tracks.shape)
    ntracks = tracks.shape[1]
    if DEBUG: print("rank: ", myrank, ", ntracks: ", ntracks)
    if DEBUG: print("tune_length: ", tune_length)

    tunelist = {}
    # divide track processing up by processor
    tracks_per_proc = int((ntracks+commsize-1)/commsize)

    if DEBUG and myrank == 0: print("tracks per proc: ", tracks_per_proc)

    my_first_track = myrank*tracks_per_proc
    my_last_track = min( (myrank+1)*tracks_per_proc, ntracks )
    if DEBUG>1: print("proc: ", myrank,", first track: ", my_first_track,", last  track: ", my_last_track)

    for do_track in range(my_first_track, my_last_track):
    #for do_track in range(5,6):
        if DEBUG>1: print("proc: ", myrank,", working on track: ", do_track)

        # this is a file of bulk tracks.  This will contain an array
        #[turn_number, trknum, coords] with shape
        # nturns x ntracks x 7

        # interp_tunes takes data in shape 6 x <nturns>
        # data in h5 filee is <nturns> x <nparticles> x 6
        coords = tracks[:, do_track, :].transpose()
        trackid = abs(tracks[0, do_track, 6])
        if DEBUG>5: print("myrank: ", myrank, ",  coords.shape: ", coords.shape)
        nturns = coords.shape[1]

        # this next one is for forward starting at 0
        #tune_starts = range(0, nturns-tune_length+1, tune_steps)

        # this next is ending at the last tune
        tune_starts = list(range(nturns-tune_length, -1, -tune_steps))
        tune_starts.reverse()

        if DEBUG>5: print("tune_starts: ", tune_starts)
        xtunes = []
        ytunes = []
        ltunes = []
        for tstart in tune_starts:
            if DEBUG>2:
                print("proc: ", myrank, ", track: ", do_track, ", turn: ", tstart)
            #tunes = interp_tunes(coords[:, tstart:tstart+1024])
            if DEBUG > 6:
                print('passed to interp_tunes: coords[:, tstart:tstart+tune_length].shape: ', coords[:, tstart:tstart+tune_length].shape)
            #tunes = interp_tunes(coords[:, tstart:tstart+tune_length], filter=[[0.000001, 0.0236826],[0.000001],[0.000001]])
            #tunes = interp_tunes(coords[:, tstart:tstart+tune_length])
            tunes = interp_tunes(coords[:, tstart:tstart+tune_length])
            #tunes = basic_tunes(coords[:, tstart:tstart+128])
            if DEBUG>6:
                print('tunes: ', tunes)
            xtunes.append(tunes[0])
            ytunes.append(tunes[1])
            ltunes.append(tunes[2])

        tunelist[int(trackid)] = (tune_starts, xtunes, ytunes, ltunes)
        #tunelist[trknum] = basic_tunes(coords)
        #tunelist[trknum] = cft_tunes(coords)
        if DEBUG>1: print("tunes for particle ", do_track,": ", tunelist[trackid])
                                                                    
    h5f.close()
    # send my tunes to rank 0 for writing out
 
    # rank 0 will collect all the tune data and write it out
    if myrank == 0:
        for r in range(1,commsize):
            rtunes = MPI.COMM_WORLD.recv(source=r)
            if DEBUG: print("Receiving tune data for %d tracks from rank %d"%(len(rtunes),r))
            tunelist.update(rtunes)

        if myrank == 0: print("Saving data for ",len(tunelist), " particles")
        np.save(tunes_file, tunelist)
    else:
        # send my data to rank 0
        MPI.COMM_WORLD.send(tunelist,dest=0)
