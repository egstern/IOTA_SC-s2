#!/usr/bin/env python

from math import pi
import synergia_workflow

opts = synergia_workflow.Options("foborodobo32_accel")

opts.add("seed", 12345791, "Pseudorandom number generator seed", int)

opts.add("kinetic_energy", 0.00250, "Beam kinetic energy [GeV]")
opts.add("aperture_radius", 0.050/2, "Radius of beam pipe aperture [m]")
opts.add("harmonic_number", 4, "Harmonic number of RF cavity")
opts.add("current", 0.1, 'beam current [mA]')
opts.add("matching", "6dmoments", "matching procedure 6dmoments|uniform")

opts.add("emitx", 4.3e-6, "unnormalized x RMS emittance [m-rad]")
opts.add("emity", 3.0e-6, "unnormalized y RMS emittance [m-rad]")
opts.add("std_dpop", 2.1e-3, "RMS dp/p spread")
opts.add("dpop_cutoff", 2.5, "Cutoff on dp/p in sigma")
opts.add("transverse_cutoff", 3.0, "Cutoff on transverse distributions in sigma")

opts.add("macroparticles", 1048576, "number of macro particles")

opts.add("turns", 1024, "number of turns")

opts.add('proptype', 'chef_propagate', 'Type of propagation to use {chef_propagate|libff}')

opts.add("spacecharge", None, "space charge [off|2d-openhockney|2d-bassetti-erskine|3d-openhockney", str)
opts.add("gridx", 32, "x grid size")
opts.add("gridy", 32, "y grid size")
opts.add("gridz", 128, "z grid size")

opts.add("stepper", "elements", "which stepper to use independent|elements|splitoperator|splitoperator_elements")
opts.add("steps", 1, "# steps")

opts.add("tracks", 100, "number of particles to track")
opts.add("save_particles", 1, "if non-zero, num particles to save")
opts.add("particles_period", 25, "save  particles every n turns")

job_mgr = synergia_workflow.Job_manager("iota.py", opts, ["iota_options.py","machine.seq"])
