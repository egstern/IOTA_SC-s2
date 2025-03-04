#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import numpy as np
import synergia
PCONST = synergia.foundation.pconstants

from iota_options import opts

#####################################

# quick and dirty twiss parameter calculator from 2x2 courant-snyder map array
def map2twiss(csmap):
    cosmu = 0.5 * (csmap[0,0]+csmap[1,1])
    asinmu = 0.5*(csmap[0,0]-csmap[1,1])

    if abs(cosmu) > 1.0:
        raise RuntimeError("map is unstable")

    mu =np.arccos(cosmu)

    # beta is positive
    if csmap[0,1] < 0.0:
        mu = 2.0 * np.pi - mu

    beta = csmap[0,1]/np.sin(mu)
    alpha = asinmu/np.sin(mu)
    tune = mu/(2.0*np.pi)

    return (alpha, beta, tune)

################################################################################

def print_bunch_stats(bunch, fo):
    coord_names = ("x", "xp", "y", "yp", "c*dt", "dp/p")

    means = synergia.bunch.Core_diagnostics().calculate_mean(bunch)
    stds = synergia.bunch.Core_diagnostics().calculate_std(bunch, means)
    print("%20s   %20s   %20s"%("coord","mean","rms"), file=fo)
    print("%20s   %20s   %20s"%("====================",
                                      "====================",
                                      "===================="), file=fo)
    for i in range(6):
        print("%20s   %20.12e   %20.12e"%(coord_names[i], means[i], stds[i]), file=fo)


################################################################################
################################################################################

# determine the bunch charge from beam current
def beam_current_to_numpart(current, length, beta, harmonic):
    rev_time = length/(beta*PCONST.c)
    total_charge = current*rev_time/PCONST.e
    # charge is divided into bunches by harmonoic number
    return total_charge/harmonic
    

################################################################################
################################################################################

def get_lattice():
    # read the lattice in from a MadX sequence file
    lattice = synergia.lattice.MadX_reader().get_lattice("iota", "machine.seq")

    # The sequence doesn't have a reference particle so define it here
    KE = opts.kinetic_energy
    mass = PCONST.mp

    etot = KE + mass
    refpart = synergia.foundation.Reference_particle(1, mass, etot)

    lattice.set_reference_particle(refpart)
    # Change the tune of one plane to break the coupling resonance
    # for elem in lattice.get_elements():
    #     if elem.get_type() == ET.quadrupole:
    #         k1 = elem.get_double_attribute('k1')
    #         elem.set_double_attribute('k1', k1*0.99)
    #         break

    if opts.proptype == "chef_propagate":
        lattice.set_all_string_attribute('extractor_type', 'chef_propagate')
    elif opts.proptype == 'libff':
        lattice.set_all_string_attribute('extractor_type', 'libff')
    else:
        lattice.set_all_string_attribute('extractor_type', 'foobar')

    return lattice

################################################################################

def set_apertures(lattice):
    for elem in lattice.get_elements():
        elem.set_string_attribute('aperture_type', 'circular')
        elem.set_double_attribute('circular_aperture_radius', opts.aperture_radius)
        
################################################################################

# Get IOTA lattice functions
def get_lattice_fns(lattice_simulator):
    lattice = lattice_simulator.get_lattice()
    return lattice_simulator.get_lattice_functions(lattice.get_elements()[-1])

################################################################################

# fill the 102 spectator particles in the bunch
def fill_spectator_particles(bunch, stdx, stdy):
    comm = bunch.get_comm()
    mpisize = comm.get_size()
    myrank = comm.get_rank()

    local_sp = bunch.get_local_spectator_particles()

    all_spectator = np.zeros((102, 6), dtype='d')
    # first 51 are intervals of stdx/10 up to 5*stdx
    all_spectator[0:51, 0] = stdx*np.arange(51)/10
    # give a small nonzero offset so the fft gives a sensible value
    all_spectator[0, 0] = 1.0e-8
    # 2nd 51 are intervals of stdy/10
    all_spectator[51:102, 2] = stdy*np.arange(51)/10
    all_spectator[51, 2] = 1.0e-8

    offsets,counts = synergia.utils.decompose_1d_raw(mpisize, 102)
    #  copy in spectator particles corresponding to this rank
    local_sp[:counts[myrank], 0:6] = all_spectator[offsets[myrank]:offsets[myrank]+counts[myrank], 0:6]


################################################################################

logger = synergia.utils.Logger(0)

# read the lattice in from a MadX sequence file
lattice = get_lattice()
set_apertures(lattice)

refpart = lattice.get_reference_particle()

energy = refpart.get_total_energy()
momentum = refpart.get_momentum()
gamma = refpart.get_gamma()
beta = refpart.get_beta()

print("energy: ", energy, file=logger)
print("momentum: ", momentum, file=logger)
print("gamma: ", gamma, file=logger)
print("beta: ", beta, file=logger)

lattice_length = lattice.get_length()
print("lattice length: ", lattice_length, file=logger)

stepper = synergia.simulation.Independent_stepper_elements(lattice, 1, 1)
# the lattice_simulator object lets us do some computations for
# lattice functions and other parameters.
lattice_simulator = stepper.get_lattice_simulator()

lattice_functions = get_lattice_fns(lattice_simulator)

lattice_simulator.register_closed_orbit()

f = open("iota_lattice.out", "w")
print(lattice.as_string(), file=f)
f.close()

myrank = 0
map = lattice_simulator.get_linear_one_turn_map()
print("one turn map from synergia2.5 infrastructure", file=logger)
print(np.array2string(map, max_line_width=200), file=logger)

[l, v] = np.linalg.eig(map)

#print( "l: ", l)
#print( "v: ", v)

print("eigenvalues: ", file=logger)
for z in l:
    print("|z|: ", abs(z), " z: ", z, " tune: ", np.log(z).imag/(2.0*np.pi), file=logger)

[ax, bx, qx] = map2twiss(map[0:2,0:2])
[ay, by, qy] = map2twiss(map[2:4, 2:4])
[az, bz, qz] = map2twiss(map[4:6,4:6])

print("Lattice parameters (assuming uncoupled map)", file=logger)
print("alpha_x: ", ax, " alpha_y: ", ay, file=logger)
print("beta_x: ", bx, " beta_y: ", by, file=logger)
print("q_x: ", qx, " q_y: ", qy, file=logger)
print("q_z: ", qz, " beta_z: ", bz, file=logger)

alpha_x = lattice_functions.alpha_x
beta_x = lattice_functions.beta_x
alpha_y = lattice_functions.alpha_y
beta_y = lattice_functions.beta_y
Dx = lattice_functions.D_x

print('Lattice parameters from CHEF', file=logger)
print('beta_x: ', beta_x, ', alpha_x: ', alpha_x, file=logger)
print('Dx: ', Dx, file=logger)
print('beta_y: ', beta_y, ', alpha_y: ', alpha_y, file=logger)

#lattice_simulator.print_lattice_functions()

alpha_c = lattice_simulator.get_momentum_compaction()
slip_factor = alpha_c - 1/gamma**2
print("alpha_c: ", alpha_c, ", slip_factor: ", slip_factor, file=logger)

hchrom = lattice_simulator.get_horizontal_chromaticity()
vchrom = lattice_simulator.get_vertical_chromaticity()

print("horizontal chromaticity: %.16g"%hchrom, file=logger)
print("vertical chromaticity: %.16g"%vchrom, file=logger)

chef_beamline = lattice_simulator.get_chef_lattice().get_beamline()
f = open("iota_beamline.out","w")
print(synergia.lattice.chef_beamline_as_string(chef_beamline), file=f)
f.close()

macro_particles = opts.macroparticles

print("macro_particles: ", macro_particles, file=logger)

comm = synergia.utils.Commxx()

bunch_charge = beam_current_to_numpart(opts.current, lattice_length, beta, opts.harmonic_number)
print('beam current: ', opts.current, ' mA', file=logger)
print('bunch created with ', opts.macroparticles, ' macroparticles', file=logger)
print('bunch charge: ', bunch_charge, file=logger)

# create matched distribution
stdx = np.sqrt(opts.emitx * beta_x + (opts.std_dpop**2)*(Dx**2))
stdy = np.sqrt(opts.emity * beta_y)
stddpop = opts.std_dpop

# generate a 6D matched bunch using either normal forms or a 6D moments procedure
if opts.matching == "6dmoments":
    print("Matching with 6d moments", file=logger)
    #bunch = synergia.bunch.Bunch(refpart, macro_particles, 102, bunch_charge, comm)
    bunch = synergia.bunch.Bunch(refpart, macro_particles, bunch_charge, comm)
    
    corr_matrix = synergia.bunch.get_correlation_matrix(map, stdx, stdy, np.float64(opts.std_dpop), np.float64(beta), [0, 2, 5])

    seed = opts.seed
    dist = synergia.foundation.Random_distribution(seed, comm)

    means = np.zeros(6, dtype='d')
    tc = opts.transverse_cutoff
    lc = opts.dpop_cutoff
    limits = np.array([tc, tc, tc, tc, lc, lc])
    synergia.bunch.populate_6d_truncated(dist, bunch, means, corr_matrix, limits)
elif opts.matching == "normalform":
    print("Matching with normal form", file=logger)
    actions = lattice_simulator.get_stationary_actions(stdx, stdy, stdz/beta)
    bunch = synergia.bunch.Bunch(refpart, macro_particles, bunch_charge, comm)
    seed = opts.seed
    dist = synergia.foundation.Random_distribution(seed, comm)
    synergia.simulation.populate_6d_stationary_gaussian(dist, bunch, actions, lattice_simulator)
else:
    bunch = synergia.bunch.Bunch(refpart, opts.macroparticles, bunch_charge, comm)
    lp = bunch.get_local_particles()
    localnum = bunch.get_local_num()
    lp[:localnum, 0:6] = 0.0

print_bunch_stats(bunch, logger)

#fill_spectator_particles(bunch, stdx, stdy)

bunch_simulator = synergia.simulation.Bunch_simulator(bunch)

# define the bunch diagnostics to save

bunch_simulator.add_per_turn(synergia.bunch.Diagnostics_full2("diag.h5"))
if opts.tracks:
    print("saving ", opts.tracks, " particle tracks", file=logger)
    bunch_simulator.add_per_turn(synergia.bunch.Diagnostics_bulk_track("tracks.h5", opts.tracks))
if opts.save_particles:
    bunch_simulator.add_per_turn(synergia.bunch.Diagnostics_particles("particles.h5"), opts.particles_period)

# save tracks for the spectators
bunch_simulator.add_per_turn(synergia.bunch.Diagnostics_bulk_spectator_track("spec_tracks.h5", 102))

# define the stepper, propagator and run the simulation.

steppertype = opts.stepper
if opts.spacecharge and not (steppertype == "splitoperator" or steppertype == "splitoperatorelements"):
    print("changing stepper to splitoperator because spacecharge is ON", file=logger)
    steppertype = "splitoperator"

if opts.spacecharge is None:
    spacecharge = None
    coll_operator = synergia.simulation.Dummy_collective_operator("foo")
    print("space charge is OFF", file=logger)
elif ((opts.spacecharge == "off") or (opts.spacecharge == "0")):
    spacecharge = None
    coll_operator = synergia.simulation.Dummy_collective_operator("foo")
    print("space charge is OFF", file=logger)
elif opts.spacecharge == "2d-bassetti-erskine":
    print("space charge 2d-bassetti-erskine is ON", file=logger)
    coll_operator = synergia.collective.Space_charge_2d_bassetti_erskine()
    coll_operator.set_longitudinal(0)
elif opts.spacecharge == "2d-openhockney":
    print("space charge 2d-openhockney is ON", file=logger)
    # openhockney space charge requires a communicator for collective effects
    coll_comm = synergia.utils.Commxx(True)
    print("space charge grid: ", opts.gridx, opts.gridy, opts.gridz, file=logger)
    grid = [opts.gridx, opts.gridy, opts.gridz]
    coll_operator = synergia.collective.Space_charge_2d_open_hockney(coll_comm, grid)
elif opts.spacecharge == "3d-openhockney":
    print("space charge 3d-openhockney is ON", file=logger)
    coll_comm = synergia.utils.Commxx(True)
    print("space charge grid: ", opts.gridx, opts.gridy, opts.gridz, file=logger)
    grid = [opts.gridx, opts.gridy, opts.gridz]
    coll_operator = synergia.collective.Space_charge_3d_open_hockney(coll_comm, grid)
else:
    raise RuntimeError("unknown space charge operator")


print(comm.get_rank(), " propagating bunch with ", bunch.get_local_num(), " particles", file=logger)


if steppertype == "independent":
    stepper = synergia.simulation.Independent_stepper(lattice, 1, opts.steps)
    print(f"using independent stepper, {opts.steps} steps/turn", file=logger)
elif steppertype == "elements":
    stepper = synergia.simulation.Independent_stepper_elements(lattice, 1, opts.steps)
    print(f"using independent stepper elements, {opts.steps} steps/element", file=logger)
elif steppertype == "splitoperator":
    stepper = synergia.simulation.Split_operator_stepper(lattice, 1, coll_operator, opts.steps)
    print(f"using split operator stepper, {opts.steps} steps/turn", file=logger)
elif steppertype == "splitoperatorelements":
    stepper = synergia.simulation.Split_operator_stepper_elements(lattice, 1, coll_operator, opts.steps)
    print(f"using split operator stepper elements, {opts.steps} steps/element", file=logger)
else:
    raise RuntimeError(f'unknown stepper requested: ->{steppertype}<-')

propagator = synergia.simulation.Propagator(stepper)
propagator.set_checkpoint_period(100)

#propagator.propagate(bunch_simulator, 100, 100, 1)
# propagator.propagate(bunch_simulator, turns, max_turns, verbosity)
propagator.propagate(bunch_simulator, opts.turns, opts.turns, 1)
