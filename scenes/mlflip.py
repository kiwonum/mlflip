# ----------------------------------------------------------------------------
#
# MantaFlow fluid solver framework
# Copyright 2018 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# MLFLIP
#
# ----------------------------------------------------------------------------

import os, sys, argparse, pickle

parser = argparse.ArgumentParser(description='MLFLIP', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(      '--nogui',  action='store_true',  help='no GUI')
parser.add_argument(      '--pause',  action='store_true',  help='pause')
parser.add_argument(      '--load', default=None,           help='path to the trained tensorflow model directory')
parser.add_argument('-w', '--window', default=1, type=int,  help='window size for sampling features; 1 (default) means 3^D, 2 means 5^D, so on.')
parser.add_argument(      '--hs', default=1.0, type=float,  help='spacing for stencil')
parser.add_argument(      '--ht', default=1.0, type=float,  help='temporal scaling for decision; 1 means the timestep per frame')
parser.add_argument(      '--th', default=None, type=float, help='decision threshold')
parser.add_argument(      '--noschk', action='store_true',  help='turn off the sanity check process')
parser.add_argument(      '--novsp',  action='store_true',  help='turn off the sampling per cell')
parser.add_argument(      '--novar',  action='store_true',  help='turn off the variance process')
parser.add_argument(      '--ongeom', action='store_true',  help='store geom values into input feature vector')
pargs = parser.parse_known_args()[0]

if pargs.load is None: sys.exit('You have to specify the path to the trained model.')
pargs.load = os.path.normpath(pargs.load)

tfopt = pickle.load(open(pargs.load + '/run_args.pickle', 'rb'))
scale = pickle.load(open(pargs.load + '/scale.pickle', 'rb'))

# default solver parameters
params               = {}
params['dim']        = 2                  # dimension
params['sres']       = 2                  # sub-resolution per cell
params['dx']         = 1.0/params['sres'] # particle spacing (= 2 x radius)
params['res']        = 50                 # reference resolution
params['len']        = 1                  # reference length
params['bnd']        = 3                  # boundary cells
params['gref']       = -9.8               # real-world gravity
params['cgaccuracy'] = 1e-3               # cg solver's threshold
params['jitter']     = 0.5                # jittering particles
params['gfm']        = True               # 2nd order fluid-empty BC
params['fps']        = 30                 # frames per second
params['t_end']      = 5.0                # quit simulation
params['sdt']        = None               # fix timestep size

# scale unit in regard to the manta world
scaleToManta   = float(params['res'])/params['len']
params['gs']   = [round(float(params['res'])*3.2)+params['bnd']*2, params['res']*3+params['bnd']*2, params['res']+params['bnd']*2 if params['dim']==3 else 1]
params['grav'] = params['gref']*scaleToManta

import numpy as np
dtype_real = np.float32         # NOTE: if double precision, use float64
dtype_int  = np.int32           # NOTE: if int in C is 64bits, use int64
np.random.seed(1)

import tensorflow as tf
tf_sess = tf.InteractiveSession()

import tf_network
dlayers = list(map(int, tfopt['dnet'].split('-')))
mlayers = list(map(int, tfopt['mnet'].split('-')))
dact    = list(map(tf_network.parse_act, tfopt['dact'].split('-')))
mact    = list(map(tf_network.parse_act, tfopt['mact'].split('-')))
x       = tf.placeholder(tf.float32, shape=[None, dlayers[0]], name='x-input')
y_,  y  = tf_network.build_network(dlayers, dact, input_x_holder=x, bn=True, is_training=False, scope='detector/')[1:]
y2_, y2 = tf_network.build_network(mlayers, mact, input_x_holder=x, bn=True, is_training=False, scope='modifier/')[1:]
if tfopt['mve']:
    sd  = tf_network.build_network(mlayers, mact, input_x_holder=x, input_y_holder=y2_, bn=True, is_training=False, scope='modifier_var/')[2]

if (not tfopt['nosmax']): y = tf.nn.softmax(y)

tf_saver = tf.train.Saver()
tf_saver.restore(tf_sess, pargs.load + '/model.ckpt')

s = Solver(name="MLFLIP", gridSize=vec3(params['gs'][0], params['gs'][1], params['gs'][2]), dim=params['dim'])
s.cfl         = 1
s.frameLength = 1.0/float(params['fps'])
s.timestepMin = 0
s.timestepMax = s.frameLength
s.timestep    = s.frameLength

# prepare grids and particles
gFlags   = s.create(FlagGrid)
gV       = s.create(MACGrid)
gVold    = s.create(MACGrid)
gP       = s.create(RealGrid)
gPhiSld  = s.create(LevelsetGrid)
gFlagTmp = s.create(FlagGrid)

pp    = s.create(BasicParticleSystem)
pT    = pp.create(PdataInt)
pV    = pp.create(PdataVec3)
pVtmp = pp.create(PdataVec3)
pVtm2 = pp.create(PdataVec3)
pVtm3 = pp.create(PdataVec3)
pItmp = pp.create(PdataInt)

gPhi    = s.create(LevelsetGrid)
gIdxSys = s.create(ParticleIndexSystem)
gIdx    = s.create(IntGrid)

mesh = s.create(name='mesh', type=Mesh) if (params['dim']==3 and not pargs.nogui) else None

fv_N_axis = 2*pargs.window + 1
fv_N_stn  = fv_N_axis*fv_N_axis*(fv_N_axis if params['dim']==3 else 1)
fv_N_row  = params['dim']*fv_N_stn + fv_N_stn + (fv_N_stn if pargs.ongeom else 0)
fv_vscale = params['len']/float(params['res']) # to the real world scale

ml_timestep = pargs.ht/params['fps']
ml_timer = 0.0

# boundary setup
gFlags.initDomain(params['bnd']-1)
bndBox = s.create(Box, p0=vec3(0), p1=vec3(params['gs'][0], params['gs'][1], params['gs'][2]))
inBox  = s.create(Box, p0=vec3(params['bnd'], params['bnd'], params['bnd'] if params['dim']==3 else 0), p1=vec3(params['gs'][0]-params['bnd'], params['gs'][1]-params['bnd'], (params['gs'][0]-params['bnd']) if params['dim']==3 else 1))
gPhiSld.join(bndBox.computeLevelset(notiming=True), notiming=True)
gPhiSld.subtract(inBox.computeLevelset(notiming=True), notiming=True)

# obstacle
a   = vec3(0.744*scaleToManta+params['bnd'], 0.161*0.5*scaleToManta+params['bnd'], 0.5*params['gs'][2] if (params['dim']==3) else 0)
b   = vec3(0.161*0.5*scaleToManta, 0.161*0.5*scaleToManta, 0.403*0.5*scaleToManta if (params['dim']==3) else params['gs'][2])
obs = s.create(Box, center=a, size=b)
obs.applyToGrid(grid=gFlags, value=FlagObstacle, respectFlags=gFlags)
gPhiSld.join(obs.computeLevelset(notiming=True), notiming=True)

# fluid setup: dam
dam_c = [2.606, 0.275, 0.5]
dam_s = [1.228*0.5, 0.55*0.5, 0.5]
a     = vec3(dam_c[0]*scaleToManta+params['bnd'], dam_c[1]*scaleToManta+params['bnd'], dam_c[2]*scaleToManta+params['bnd'] if (params['dim']==3) else 0)
b     = vec3(dam_s[0]*scaleToManta, dam_s[1]*scaleToManta, dam_s[2]*scaleToManta if (params['dim']==3) else params['gs'][2])
fld   = s.create(Box, center=a, size=b)
fld.applyToGrid(grid=gFlags, value=FlagFluid, respectFlags=gFlags)

begin = pp.size()
sampleShapeWithParticles(shape=fld, flags=gFlags, parts=pp, discretization=params['sres'], randomness=0, notiming=True)
end = pp.size()
pT.setConstRange(s=FlagFluid, begin=begin, end=end, notiming=True)

paramMarkFluidCell = dict(parts=pp, flags=gFlags, ptype=pT, exclude=FlagObstacle|FlagOpen)
paramSolvePressure = dict(flags=gFlags, vel=gV, pressure=gP, cgAccuracy=params['cgaccuracy'])
paramSetWallBcs    = dict(flags=gFlags, vel=gV)

if params['gfm']:               # for the free-surface boundary condition
    paramSolvePressure.update(phi=gPhi)

if not pargs.nogui:
    gui = Gui()
    gui.show()
    if pargs.pause: gui.pause()

expected  = np.zeros(pp.size(), dtype=dtype_real)
while (s.timeTotal<params['t_end']): # main loop
    mapPartsToMAC(vel=gV, flags=gFlags, velOld=gVold, parts=pp, partVel=pV, ptype=pT, exclude=FlagOpen|FlagEmpty)

    if params['sdt'] is None: s.adaptTimestep(gV.getMax())
    else: s.adaptTimestepByDt(params['sdt'])

    addGravityNoScale(flags=gFlags, vel=gV, gravity=vec3(0, params['grav'], 0))

    gridParticleIndex(parts=pp, flags=gFlags, indexSys=gIdxSys, index=gIdx)
    unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=gPhi, radiusFactor=1.0)
    extrapolateLsSimple(phi=gPhi, distance=max(4, int(4*pargs.hs)), inside=True)
    extrapolateLsSimple(phi=gPhi, distance=max(4, int(4*pargs.hs)), inside=False)

    if (params['dim']==3 and not pargs.nogui):
        gPhi.createMesh(mesh)

    setWallBcs(**paramSetWallBcs)
    solvePressure(**paramSolvePressure)
    setWallBcs(**paramSetWallBcs)
    extrapolateMACSimple(flags=gFlags, vel=gV)

    # BEGIN: machine learning part >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # get candidate particles
    gFlagTmp.copyFrom(gFlags)
    gFlagTmp.extendRegion(region=FlagEmpty, exclude=FlagObstacle, depth=1)
    pItmp.copyFrom(pT)
    pp.setType(ptype=pItmp, mark=0, stype=FlagEmpty, flags=gFlagTmp, cflag=FlagEmpty|FlagFluid) # already individual? then, kill
    pp.setType(ptype=pItmp, mark=FlagEmpty, stype=FlagFluid, flags=gFlagTmp, cflag=FlagEmpty)   # mark surface particles
    candidate = np.zeros(pp.size(), dtype=dtype_int)
    copyPdataToArrayInt(target=candidate, source=pItmp)
    candidate = (candidate==FlagEmpty)
    N_candidate = np.count_nonzero(candidate)

    # extract features -> numpy array
    inputs_c = np.zeros(pp.size()*fv_N_row, dtype=dtype_real)
    off_feature = 0
    extractFeatureVelRel(fv=inputs_c, N_row=fv_N_row, off_begin=off_feature, p=pp, vel=gV, scale=fv_vscale, ptype=pItmp, exclude=FlagObstacle|FlagFluid, window=pargs.window, h=pargs.hs)
    off_feature = params['dim']*fv_N_stn

    extractFeaturePhi(fv=inputs_c, N_row=fv_N_row, off_begin=off_feature, p=pp, phi=gPhi, scale=1.0, ptype=pItmp, exclude=FlagObstacle|FlagFluid, window=pargs.window)
    off_feature += fv_N_stn

    if pargs.ongeom:
        extractFeatureGeo(fv=inputs_c, N_row=fv_N_row, off_begin=off_feature, p=pp, flag=gFlags, scale=1.0, ptype=pItmp, exclude=FlagObstacle|FlagFluid, window=pargs.window)
        off_feature += fv_N_stn

    inputs_c = inputs_c.reshape((-1, fv_N_row))[candidate]

    # evaluate using the networks

    # run tf: detection and modification
    if tfopt['mve']:    dtct_c, dv_c, appx_s_c = tf_sess.run([y, y2, sd], feed_dict={x: inputs_c})
    else:               dtct_c, dv_c           = tf_sess.run([y, y2],     feed_dict={x: inputs_c})
    if not tfopt['nosmax']: dtct_c = np.delete(dtct_c, 1, -1)

    # update expected value
    expected_c = np.zeros(N_candidate, dtype=dtype_real)
    expected_c = 2.0*dtct_c.reshape(-1) - 1.0
    scale_time = 1.0/np.sqrt(s.frameLength/s.timestep)
    expected[candidate] += expected_c*scale_time
    expected[np.invert(candidate)] -= 1.0*scale_time

    # END: machine learning part <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # update velocity; general update from FLIP
    flipVelocityUpdate(vel=gV, velOld=gVold, flags=gFlags, parts=pp, partVel=pV, flipRatio=0.97, ptype=pT, exclude=FlagObstacle|FlagOpen|FlagEmpty)

    # update velocity; individual update for Lagrangian particles
    pp.updateVelocity(vel=pV, a=vec3(0, params['grav'], 0), dt=s.timestep, ptype=pT, exclude=FlagObstacle|FlagFluid)

    pp.getPosPdata(target=pVtmp) # save the old position for ballistic handling

    # splash decision per frame
    if ml_timer <= 0:
        decision = np.zeros(pp.size(), dtype=dtype_int)
        decision[(expected>0)] = FlagEmpty
        decision[(decision!=FlagEmpty)] = FlagFluid

        if params['dim']==2:
            dv_c = np.append(dv_c, np.zeros((N_candidate, 1), dtype=dtype_real), axis=1)
            if tfopt['mve']: appx_s_c = np.append(appx_s_c, np.zeros((N_candidate, 1), dtype=dtype_real), axis=1)

        if tfopt['mve']:
            appx_s_c = np.fabs(appx_s_c)
            if not pargs.novar: dv_c += appx_s_c*np.random.normal(size=(N_candidate, 3))

        dv = np.zeros((pp.size(), 3), dtype=dtype_real)
        dv[candidate] = dv_c*scale['modvel']/fv_vscale

        if not pargs.noschk:
            # 0. let's predict splash in framestep, which is the same with the training data's timestep
            c_dt = s.timestep
            s.timestep = s.frameLength

            # 1. mark fluid region after moving without splash-correction
            gFlagTmp.copyFrom(gFlags)
            pp.advectInGrid(flags=gFlagTmp, vel=gV, integrationMode=IntRK4, deleteInObstacle=False, ptype=pT, exclude=FlagObstacle|FlagOpen|FlagEmpty)
            pp.advect(vel=pV, ptype=pT, exclude=FlagObstacle|FlagFluid)
            markFluidCells(parts=pp, flags=gFlagTmp, ptype=pT, exclude=FlagObstacle|FlagOpen|FlagEmpty)
            # gFlagTmp.extendRegion(region=FlagEmpty, exclude=FlagObstacle, depth=1) # let's allow splash particles lay on the surface

            # 2. try to move splashing particles only, so check if it's really splashing; revert the wrong decisions
            pp.setPosPdata(source=pVtmp)
            pVtm2.copyFrom(pV)
            dv[(decision!=FlagEmpty)] = 0
            copyArrayToPdataVec3(target=pVtm3, source=dv.reshape(-1, 1))
            pVtm2.add(pVtm3)
            copyArrayToPdataInt(target=pItmp, source=decision)
            pp.advect(vel=pVtm2, ptype=pItmp, exclude=FlagObstacle|FlagFluid|FlagOpen)
            pp.setType(ptype=pItmp, mark=FlagFluid, stype=FlagEmpty, flags=gFlagTmp, cflag=FlagFluid|FlagObstacle) # empty -> fluid if they are not acturally splashing
            copyPdataToArrayInt(target=decision, source=pItmp)

            # 3. roll-back
            s.timestep = c_dt
            pp.setPosPdata(source=pVtmp)

        if not pargs.novsp:
            # final decision and velocity modification using sampling (NOTE: currently, the ratio is one particle per cell)
            getGridIdx(gidx=pItmp, parts=pp, grid=gFlags)
            grididx = np.zeros(pp.size(), dtype=dtype_int)
            copyPdataToArrayInt(target=grididx, source=pItmp)
            gidx2pidxs = {}
            idx = np.argwhere(decision==FlagEmpty).reshape(-1)
            for pidx in idx:
                if ((grididx[pidx] not in gidx2pidxs) or (expected[gidx2pidxs[grididx[pidx]]] < expected[pidx])):
                    gidx2pidxs[grididx[pidx]] = pidx

            idx = np.array(list(gidx2pidxs.values()), dtype=dtype_int).reshape(-1)
            decision = np.zeros(pp.size(), dtype=dtype_int)
            decision[idx] = FlagEmpty
            decision[(decision!=FlagEmpty)] = FlagFluid

        # mark splashing particles and modify their velocities so that they can flow individually
        copyArrayToPdataInt(target=pItmp, source=decision)
        pT.setConstIntFlag(s=FlagEmpty, t=pItmp, itype=FlagEmpty)
        dv[(decision!=FlagEmpty)] = 0
        copyArrayToPdataVec3(target=pVtm2, source=dv.reshape(-1, 1))
        pV.add(pVtm2)

        expected = np.zeros(pp.size(), dtype=dtype_real) # clear expected values for the next frame
        ml_timer = ml_timestep

    # update position
    pp.advectInGrid(flags=gFlags, vel=gV, integrationMode=IntRK4, deleteInObstacle=False, ptype=pT, exclude=FlagObstacle|FlagOpen|FlagEmpty)
    pp.advect(vel=pV, ptype=pT, exclude=FlagObstacle|FlagFluid)
    pp.projectOutOfBnd(flags=gFlags, bnd=params['bnd']+params['dx']*0.5, plane='xXyYzZ', ptype=pT, exclude=FlagObstacle)
    pushOutofObs(parts=pp, flags=gFlags, phiObs=gPhiSld, thresh=params['dx']*0.5, ptype=pT, exclude=FlagObstacle)

    # update velocity of the Lagrangian particles
    pp.updateVelocityFromDeltaPos(vel=pV, x_prev=pVtmp, dt=s.timestep, ptype=pT, exclude=FlagObstacle|FlagFluid)

    # handling particles going out or coming into the simulation domain
    pp.setType(ptype=pT, mark=FlagOpen, stype=FlagFluid|FlagEmpty, flags=gFlags, cflag=FlagOpen)  # mark as open
    pp.setType(ptype=pT, mark=FlagFluid, stype=FlagOpen, flags=gFlags, cflag=FlagEmpty|FlagFluid) # mark as fluid
    markFluidCells(**paramMarkFluidCell)

    pp.setType(ptype=pT, mark=FlagFluid, stype=FlagEmpty, flags=gFlags, cflag=FlagFluid) # empty -> fluid if they enter again.

    # NOTE: We don't need to solve the pressure for isolated cells.
    markIsolatedFluidCell(flags=gFlags, mark=FlagEmpty)
    pp.setType(ptype=pT, mark=FlagEmpty, stype=FlagFluid, flags=gFlags, cflag=FlagEmpty) # fluid -> empty if they escape

    # keep the valid splashing judgements; the particles may still stay inside the flow (due to a small timestep size)
    copyArrayToPdataInt(target=pItmp, source=decision)
    pT.setConstIntFlag(s=FlagEmpty, t=pItmp, itype=FlagEmpty) # splashing (empty) judgment is still valid

    ml_timer -= s.timestep
    s.step()

tf_sess.close()
