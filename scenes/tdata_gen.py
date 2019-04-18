# ----------------------------------------------------------------------------
#
# MantaFlow fluid solver framework
# Copyright 2018 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Training data generator: Read a simulation data (particle/velocity),
# re-interpret in a target resolution, and generate training data
#
# ----------------------------------------------------------------------------

import os, argparse, glob, pickle
parser = argparse.ArgumentParser(description='Generate Training Data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(      '--guion',  action='store_true',         help='GUI')
parser.add_argument(      '--pause',  action='store_true',         help='pause')
parser.add_argument('-d', '--dscale', default=0.5, type=float,     help='target resolution (usally, for down-scaling)')
parser.add_argument('-s', '--marks',  default=4, type=int,         help='mark as splashing particles if the region is smaller than this value (in source (i.e., high-res) scale)')
parser.add_argument('-w', '--window', default=1, type=int,         help='window size for sampling features; 1 (default) means 3^D, 2 means 5^D, so on.')
parser.add_argument(      '--hs',     default=1.0, type=float,     help='spacing for window stencil; 1 (default) means 1 cell in the target resolution')
parser.add_argument('-p', '--pfile',  default='particles.uni',     help='file name for particle data')
parser.add_argument('-v', '--vfile',  default='particlesVel.uni',  help='file name for particle velocity data')
parser.add_argument('-t', '--tfile',  default='particlesType.uni', help='file name for particle type data')
parser.add_argument('-o', '--output', default='/tmp/tdata/',       help='output path prefix for training data; will save output_tdata_p{0,1}.npz')
parser.add_argument(      '--prefv',  default='fv',                help='output path prefix for feature vector data (e.g. 00001/fv/)')
parser.add_argument(      '--ongeom', action='store_true',         help='store geom values into input feature vector')
parser.add_argument(      '--outpng', default=None,                help='output path for png files')
parser.add_argument(      'simdir',   action="store", type=str,    help='simulation path; will search for params.pickle and frames (e.g, 00000/, 00001/, and so on.)')
pargs = parser.parse_args()
pargs.output = os.path.normpath(pargs.output)

import numpy as np
dtype_real = np.float32         # NOTE: if double precision, use float64
dtype_int  = np.int32           # NOTE: if int in C is 64bits, use int64

import tf_uniio as uni

loadpath = pargs.simdir+'/params.pickle'
if pargs.simdir is None or not os.path.isfile(loadpath):
    sys.exit('Cannot load simulation: ' + loadpath)
with open(loadpath, 'rb') as f:
    params = pickle.load(f)

scaleToManta    = float(params['res'])/params['len']
params['gs']    = [params['res']+params['bnd']*2, params['res']+params['bnd']*2, params['res']+params['bnd']*2 if params['dim']==3 else 1]
params['grav']  = params['gref']*scaleToManta
params['stens'] = params['stref']*scaleToManta

# prepare grids and particles
#  accurate simulation
xl_s             = Solver(name='Re-simulate using FLIP (Original)', gridSize=vec3(params['gs'][0], params['gs'][1], params['gs'][2]), dim=params['dim'])
xl_s.cfl         = 1
xl_s.frameLength = 1.0/float(params['fps'])
xl_s.timestepMin = 0
xl_s.timestepMax = xl_s.frameLength
xl_s.timestep    = xl_s.frameLength

xl_gFlags = xl_s.create(FlagGrid)
xl_gR     = xl_s.create(IntGrid)

xl_pp = xl_s.create(BasicParticleSystem)
xl_pT = xl_pp.create(PdataInt)

# boundary setup
xl_gFlags.initDomain(params['bnd']-1)

#  downscaled simulation
params['res']  *= pargs.dscale
params['dx']   *= pargs.dscale
params['stref'] = None
params['stens'] = None
params['nuref'] = None
params['visc']  = None

scaleToManta    = float(params['res'])/params['len']
params['gs']    = [params['res']+params['bnd']*2, params['res']+params['bnd']*2, params['res']+params['bnd']*2 if params['dim']==3 else 1]
params['grav']  = params['gref']*scaleToManta

s             = Solver(name='Re-simulate using FLIP (Target)', gridSize=vec3(params['gs'][0], params['gs'][1], params['gs'][2]), dim=params['dim'])
s.cfl         = 1
s.frameLength = 1.0/float(params['fps'])
s.timestepMin = 0
s.timestepMax = s.frameLength
s.timestep    = s.frameLength

gFlags  = s.create(FlagGrid)
gR      = s.create(IntGrid)
gV      = s.create(MACGrid)
gVold   = s.create(MACGrid)
gP      = s.create(RealGrid)
gPhiSld = s.create(LevelsetGrid)

pp    = s.create(BasicParticleSystem)
pT    = pp.create(PdataInt)
pV    = pp.create(PdataVec3)
pVtmp = pp.create(PdataVec3)

# boundary setup
gFlags.initDomain(params['bnd']-1)
bndBox = s.create(Box, p0=vec3(0), p1=vec3(params['gs'][0], params['gs'][1], params['gs'][2]))
inBox  = s.create(Box, p0=vec3(params['bnd'], params['bnd'], params['bnd'] if params['dim']==3 else 0), p1=vec3(params['gs'][0]-params['bnd'], params['gs'][1]-params['bnd'], (params['gs'][0]-params['bnd']) if params['dim']==3 else 1))
gPhiSld.join(bndBox.computeLevelset(notiming=True), notiming=True)
gPhiSld.subtract(inBox.computeLevelset(notiming=True), notiming=True)

bnd       = vec3(params['bnd'], params['bnd'], params['bnd'] if params['dim']==3 else 0)
xl_drange = [ bnd, xl_s.getGridSize()-bnd ]
drange    = [ bnd, s.getGridSize()-bnd ]
factor    = calcGridSizeFactorWithRange(xl_drange[0], xl_drange[1], drange[0], drange[1])

fv_N_axis = 2*pargs.window + 1
fv_N_stn  = fv_N_axis*fv_N_axis*(fv_N_axis if params['dim']==3 else 1)
fv_N_row  = params['dim']*fv_N_stn + fv_N_stn + (fv_N_stn if pargs.ongeom else 0)
fv_vscale = params['len']/float(params['res'])

def save_features(opath):
    fv = np.zeros(pp.size()*fv_N_row, dtype=dtype_real)

    off_feature = 0
    extractFeatureVelRel(fv=fv, N_row=fv_N_row, off_begin=off_feature, p=pp, vel=gV, scale=fv_vscale, ptype=pT, exclude=FlagObstacle, window=pargs.window, h=pargs.hs)
    off_feature = params['dim']*fv_N_stn

    extractFeaturePhi(fv=fv, N_row=fv_N_row, off_begin=off_feature, p=pp, phi=gPhi, scale=1.0, ptype=pT, exclude=FlagObstacle, window=pargs.window)
    off_feature += fv_N_stn

    if pargs.ongeom:
        extractFeatureGeo(fv=fv, N_row=fv_N_row, off_begin=params['dim']*fv_N_stn, p=pp, flag=gFlags, scale=1.0, ptype=pT, exclude=FlagObstacle, window=pargs.window)
        off_feature += fv_N_stn

    fv = fv.reshape((-1, fv_N_row))
    np.savez_compressed(opath, inputs=fv)

def save_new_splashing_particles(o_path, o_t_path, i_t_curr, i_p_curr, i_p_next):
    xl_pp.load(i_p_curr)
    xl_pT.load(i_t_curr)
    markFluidCells(parts=xl_pp, flags=xl_gFlags, ptype=xl_pT, exclude=FlagObstacle)
    xl_pp.setType(ptype=xl_pT, mark=FlagFluid, stype=FlagEmpty|FlagFluid, flags=xl_gFlags, cflag=FlagFluid)

    # 1. kill already splashed particles
    getRegionalCounts(r=xl_gR, flags=xl_gFlags, ctype=FlagFluid)
    markSmallRegions(flags=xl_gFlags, rcnt=xl_gR, mark=FlagEmpty, exclude=FlagObstacle|FlagOpen, th=int(pargs.marks*1.25))
    xl_pp.setType(ptype=xl_pT, mark=0, stype=FlagFluid|FlagEmpty|FlagOpen, flags=xl_gFlags, cflag=FlagEmpty)

    # 2. mark newly splashing particles
    xl_pp.load(i_p_next)
    markFluidCells(parts=xl_pp, flags=xl_gFlags, ptype=xl_pT, exclude=FlagObstacle)
    getRegionalCounts(r=xl_gR, flags=xl_gFlags, ctype=FlagFluid)
    markSmallRegions(flags=xl_gFlags, rcnt=xl_gR, mark=FlagEmpty, exclude=FlagObstacle|FlagOpen, th=pargs.marks)
    xl_pp.setType(ptype=xl_pT, mark=FlagEmpty, stype=FlagFluid, flags=xl_gFlags, cflag=FlagEmpty)

    # 3. kill meaningless particles (inside the flow body)
    xl_gFlags.extendRegion(region=FlagEmpty, exclude=FlagObstacle, depth=1)
    xl_pp.setType(ptype=xl_pT, mark=0, stype=FlagFluid, flags=xl_gFlags, cflag=FlagFluid)

    if o_t_path: xl_pT.save(o_t_path)

    np_arr = np.zeros(xl_pp.size(), dtype=dtype_int)
    copyPdataToArrayInt(target=np_arr, source=xl_pT)
    np.savez_compressed(o_path, labels=np_arr.reshape((-1, 1)))

def save_velocity_modification(o_path, i_t_gt, i_p_next, i_p_curr, i_v_next):
    # i_p_next and i_p_curr: ground truth (high-res) scale
    # i_v_next: target (low-res) scale
    xl_pp.load(i_p_next); xl_pp.getPosPdata(pVtmp)
    xl_pp.load(i_p_curr); xl_pp.getPosPdata(pV) # NOTE: pV will be reloaded so it's safe to use; let's save the memory
    pVtmp.sub(pV); pVtmp.multConst(factor); pVtmp.multConst(vec3(1.0/s.frameLength))
    pV.load(i_v_next); pVtmp.sub(pV) # dv = (x(n+1) - x(n))/dt - v(n+1)
    pVtmp.multConst(vec3(fv_vscale)) # scale to the real-world unit
    np_arr = np.zeros(pp.size()*3, dtype=dtype_real)
    copyPdataToArrayVec3(target=np_arr, source=pVtmp)
    np_arr = np_arr.reshape((-1, 3)) if params['dim']==3 else uni.drop_zdim(np_arr.reshape((-1, 3)))

    np.savez_compressed(o_path, modvel=np_arr)

gPhi    = s.create(LevelsetGrid)
gIdxSys = s.create(ParticleIndexSystem)
gIdx    = s.create(IntGrid)
paramSolvePressure = dict(flags=gFlags, vel=gV, pressure=gP, cgAccuracy=params['cgaccuracy'])
if params['gfm']:               # for the free-surface boundary condition
    paramSolvePressure.update(phi=gPhi)

if pargs.guion:
    gui = Gui()
    gui.show()
    if pargs.pause: gui.pause()

inDir = sorted(glob.glob(pargs.simdir+'/0*'))
pargs.outpng is None or os.path.isdir(pargs.outpng) or os.makedirs(pargs.outpng)
for i, dir_i in enumerate(inDir):
    if i == len(inDir)-1: break
    print('Frame: {}'.format(dir_i))

    dir_c = os.path.normpath(dir_i)
    dir_n = os.path.normpath(inDir[i+1])

    file_p_c, file_p_n = '{}/{}'.format(dir_c, pargs.pfile), '{}/flip.{}'.format(dir_n, pargs.pfile)
    file_v_c, file_v_n = '{}/{}'.format(dir_c, pargs.vfile), '{}/flip.{}'.format(dir_n, pargs.vfile)
    file_t_c, file_t_n = '{}/{}'.format(dir_c, pargs.tfile), '{}/flip.{}'.format(dir_n, pargs.tfile)
    file_p_gt          = '{}/{}'.format(dir_n, pargs.pfile)
    file_fv            = '{}/{}/input.{}'.format(dir_n, pargs.prefv, 'inputs.npz')
    file_lb            = '{}/{}/label.{}'.format(dir_n, pargs.prefv, 'labels.npz')
    file_vm            = '{}/{}/label.{}'.format(dir_n, pargs.prefv, 'modvel.npz')

    dir_fv = '{}/{}'.format(dir_n, pargs.prefv)
    os.path.isdir(dir_fv) or os.makedirs(dir_fv)

    xl_pp.load(file_p_c)
    pp.load(file_p_c)
    pp.transformFrom(p_old=xl_pp, from_old=xl_drange[0], to_old=xl_drange[1], from_new=drange[0], to_new=drange[1])
    pV.load(file_v_c)
    pV.multConst(factor)
    pT.load(file_t_c)
    markFluidCells(parts=pp, flags=gFlags, ptype=pT, exclude=FlagObstacle|FlagOpen)

    if pargs.guion:
        gui.update()
        if pargs.outpng: gui.screenshot('{}/f{:05d}_orig.png'.format(pargs.outpng, i))

    frame_last = s.frame
    while (frame_last == s.frame):

        mapPartsToMAC(vel=gV, flags=gFlags, velOld=gVold, parts=pp, partVel=pV, ptype=pT, exclude=FlagObstacle|FlagOpen|FlagEmpty)
        s.adaptTimestepByDt(s.frameLength) # NOTE: frame-to-frame

        addGravityNoScale(flags=gFlags, vel=gV, gravity=vec3(0, params['grav'], 0))

        gridParticleIndex(parts=pp, flags=gFlags, indexSys=gIdxSys, index=gIdx)
        unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=gPhi, radiusFactor=1.0, ptype=pT, exclude=FlagObstacle)
        extrapolateLsSimple(phi=gPhi, distance=4, inside=True)
        extrapolateLsSimple(phi=gPhi, distance=4, inside=False)

        setWallBcs(flags=gFlags, vel=gV)
        solvePressure(**paramSolvePressure)
        setWallBcs(flags=gFlags, vel=gV)
        extrapolateMACSimple(flags=gFlags, vel=gV)

        # extract data for feature vector at the beginning of each frame
        save_features(opath=file_fv)

        # update velocity (general update from FLIP and individual update for Lagrangian particles)
        flipVelocityUpdate(vel=gV, velOld=gVold, flags=gFlags, parts=pp, partVel=pV, flipRatio=0.97, ptype=pT, exclude=FlagObstacle|FlagOpen|FlagEmpty)
        pp.updateVelocity(vel=pV, a=vec3(0, params['grav'], 0), dt=s.timestep, ptype=pT, exclude=FlagObstacle|FlagFluid)

        # update position
        pp.getPosPdata(target=pVtmp)
        pp.advectInGrid(flags=gFlags, vel=gV, integrationMode=IntRK4, deleteInObstacle=False, ptype=pT, exclude=FlagObstacle|FlagOpen|FlagEmpty)
        pp.advect(vel=pV, ptype=pT, exclude=FlagFluid|FlagObstacle)
        pp.projectOutOfBnd(flags=gFlags, bnd=params['bnd']+params['dx']*0.5, plane='xXyYzZ', ptype=pT, exclude=FlagObstacle)
        pushOutofObs(parts=pp, flags=gFlags, phiObs=gPhiSld, thresh=params['dx']*0.5, ptype=pT, exclude=FlagObstacle)

        # update velocity of the Lagrangian particles
        pp.updateVelocityFromDeltaPos(vel=pV, x_prev=pVtmp, dt=s.timestep, ptype=pT, exclude=FlagFluid|FlagObstacle)

        # handling particles going out or coming into the simulation domain
        pp.setType(ptype=pT, mark=FlagOpen, stype=FlagFluid|FlagEmpty, flags=gFlags, cflag=FlagOpen)
        pp.setType(ptype=pT, mark=FlagFluid, stype=FlagOpen, flags=gFlags, cflag=FlagEmpty|FlagFluid)
        markFluidCells(parts=pp, flags=gFlags, ptype=pT, exclude=FlagObstacle|FlagOpen)

        # NOTE: We don't need to solve the pressure for isolated cells.
        pp.setType(ptype=pT, mark=FlagFluid, stype=FlagEmpty, flags=gFlags, cflag=FlagFluid)
        markIsolatedFluidCell(flags=gFlags, mark=FlagEmpty)
        pp.setType(ptype=pT, mark=FlagEmpty, stype=FlagFluid, flags=gFlags, cflag=FlagEmpty)

        s.step()

    pp.save(file_p_n)
    pV.save(file_v_n)
    pT.save(file_t_n)

    # extract label (here, splash vs non-splash)
    file_t_b = '{}/{}'.format(dir_fv, 'particlesType.uni')
    save_new_splashing_particles(o_path=file_lb, o_t_path=file_t_b, i_t_curr=file_t_c, i_p_curr=file_p_c, i_p_next=file_p_gt)
    save_velocity_modification(o_path=file_vm, i_t_gt=file_lb, i_p_next=file_p_gt, i_p_curr=file_p_c, i_v_next=file_v_n)

    if pargs.guion:
        gui.update()
        if (pargs.outpng is not None):
            gui.screenshot('{}/f{:05d}_flip.png'.format(pargs.outpng, i+1))

################################################################################
# generate training data (inputs/labels) for TensorFlow
basedir = '/'.join(pargs.output.split('/')[:-1])
os.path.isdir(basedir) or os.makedirs(basedir)

set_0 = {}
set_1 = {}
opath_0 = '{}_tdata_p0.npz'.format(pargs.output) # non-splashing
opath_1 = '{}_tdata_p1.npz'.format(pargs.output) # splashing
for dir_i in inDir:
    dir_i  = os.path.normpath(dir_i)
    dir_fv = '{}/{}'.format(dir_i, pargs.prefv)

    # inputs/labels
    paths = {
        'inputs': '{}/input.{}'.format(dir_fv, 'inputs.npz'),
        'labels': '{}/label.{}'.format(dir_fv, 'labels.npz'),
        'modvel': '{}/label.{}'.format(dir_fv, 'modvel.npz')
    }

    if not all([os.path.isfile(paths[x]) for x in paths]):
        print('In {}: Incomplete set of files; skipped'.format(dir_i))
        continue

    labels  = np.load(paths['labels'])['labels']
    if np.sum((labels==FlagEmpty).astype(int))==0:
        print('In {}: No splash particle; skipped'.format(dir_i))
        continue

    del_idx = [k for k, v in enumerate(labels) if (v==FlagObstacle) or (v==0)] # {0: unknown, 1: fluid; 2: obstacle; 4: empty/splash}

    inputs = np.load(paths['inputs'])['inputs']
    inputs = np.delete(inputs, del_idx, 0)

    labels = np.delete(labels, del_idx, 0)
    labels = (labels==FlagEmpty).astype(float)
    modvel = np.load(paths['modvel'])['modvel']
    modvel = np.delete(modvel, del_idx, 0)

    p0_idx = [k for k, v in enumerate(labels) if v==0.0]
    p1_idx = [k for k, v in enumerate(labels) if v==1.0]

    set_0['inputs'] = np.concatenate((set_0['inputs'], inputs[p0_idx]), axis=0) if 'inputs' in set_0 else inputs[p0_idx]
    set_0['labels'] = np.concatenate((set_0['labels'], labels[p0_idx]), axis=0) if 'labels' in set_0 else labels[p0_idx]
    set_0['modvel'] = np.concatenate((set_0['modvel'], modvel[p0_idx]), axis=0) if 'modvel' in set_0 else modvel[p0_idx]
    set_1['inputs'] = np.concatenate((set_1['inputs'], inputs[p1_idx]), axis=0) if 'inputs' in set_1 else inputs[p1_idx]
    set_1['labels'] = np.concatenate((set_1['labels'], labels[p1_idx]), axis=0) if 'labels' in set_1 else labels[p1_idx]
    set_1['modvel'] = np.concatenate((set_1['modvel'], modvel[p1_idx]), axis=0) if 'modvel' in set_1 else modvel[p1_idx]

    print('In {}: Generated {} (p0:{}, p1:{}) tuples'.format(dir_i, labels.shape[0], len(p0_idx), len(p1_idx)))

print('\nWriting to {}: {} ... '.format(opath_0, list(map(np.shape, set_0.values()))), end='')
np.savez_compressed(opath_0, **set_0)
print('Done.')
print(  'Writing to {}: {} ... '.format(opath_1, list(map(np.shape, set_1.values()))), end='')
np.savez_compressed(opath_1, **set_1)
print('Done.')
