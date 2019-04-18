# ----------------------------------------------------------------------------
#
# MantaFlow fluid solver framework
# Copyright 2018 Kiwon Um, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Training simulation using FLIP
#
# ----------------------------------------------------------------------------

import os, math, operator, random, pickle, argparse

parser = argparse.ArgumentParser(description='Training Simulator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(      '--nogui',  action='store_true', help='no GUI')
parser.add_argument(      '--pause',  action='store_true', help='pause')
parser.add_argument(      '--seed',   default=0, type=int, help='random seed')
parser.add_argument('-o', '--output', default='/tmp/tsim_flip_00', help='output path for training simulation')
pargs = parser.parse_args()
pargs.output = os.path.normpath(pargs.output)

random.seed(pargs.seed)

# default solver parameters
params               = {}
params['dim']        = 2                  # dimension
params['sres']       = 2                  # sub-resolution per cell
params['dx']         = 1.0/params['sres'] # particle spacing (= 2 x radius)
params['res']        = 100                # reference resolution
params['len']        = 1                  # reference length
params['bnd']        = 3                  # boundary cells
params['gref']       = 0                  # real-world gravity
params['stref']      = 0.073              # surface tension
params['cgaccuracy'] = 1e-3               # cg solver's threshold
params['jitter']     = 0.5                # jittering particles
params['gfm']        = True               # 2nd order fluid-empty BC
params['fps']        = 30                 # frames per second
params['t_end']      = 3.0                # quit simulation
params['sdt']        = None               # fix timestep size
params['rr1']        = [-0.25, 0.25]
params['rr2']        = [0.05, 0.01]
params['rr3']        = [1, 3]

# scale unit in regard to the manta world
scaleToManta    = float(params['res'])/params['len']
params['gs']    = [params['res']+params['bnd']*2, params['res']+params['bnd']*2, params['res']+params['bnd']*2 if params['dim']==3 else 1]
params['grav']  = params['gref']*scaleToManta
params['stens'] = params['stref']*scaleToManta

s             = Solver(name="FLIP", gridSize=vec3(params['gs'][0], params['gs'][1], params['gs'][2]), dim=params['dim'])
s.cfl         = 1
s.frameLength = 1.0/float(params['fps'])
s.timestepMin = 0
s.timestepMax = s.frameLength if (('stens' not in params) or (params['stens'] is None)) else math.sqrt(1.0/params['stens'])
s.timestep    = s.frameLength

# prepare grids and particles
gFlags  = s.create(FlagGrid)
gV      = s.create(MACGrid)
gVold   = s.create(MACGrid)
gP      = s.create(RealGrid)
gPhiSld = s.create(LevelsetGrid)

pp    = s.create(BasicParticleSystem)
pT    = pp.create(PdataInt)
pV    = pp.create(PdataVec3)
pVtmp = pp.create(PdataVec3)

mesh = s.create(name='mesh', type=Mesh) if (params['dim']==3 and pargs.guion) else None

savingFuncs = []
savingFuncs.append([pp.save, 'particles.uni'])
savingFuncs.append([pV.save, 'particlesVel.uni'])
savingFuncs.append([pT.save, 'particlesType.uni'])

paramSolvePressure = dict(flags=gFlags, vel=gV, pressure=gP, cgAccuracy=params['cgaccuracy'])
if params['gfm']:               # for the free-surface boundary condition
    gPhi    = s.create(LevelsetGrid)
    gIdxSys = s.create(ParticleIndexSystem)
    gIdx    = s.create(IntGrid)
    paramSolvePressure.update(phi=gPhi)
    if params['stens']:
        gCurv = s.create(RealGrid)
        paramSolvePressure.update(curv=gCurv, surfTens=params['stens'])

# boundary setup
gFlags.initDomain(params['bnd']-1)
bndBox = s.create(Box, p0=vec3(0), p1=vec3(params['gs'][0], params['gs'][1], params['gs'][2]))
inBox  = s.create(Box, p0=vec3(params['bnd'], params['bnd'], params['bnd'] if params['dim']==3 else 0), p1=vec3(params['gs'][0]-params['bnd'], params['gs'][1]-params['bnd'], (params['gs'][0]-params['bnd']) if params['dim']==3 else 1))
gPhiSld.join(bndBox.computeLevelset(notiming=True), notiming=True)
gPhiSld.subtract(inBox.computeLevelset(notiming=True), notiming=True)

# fluid
def normalize(v):
    vmag = math.sqrt(sum(v[i]*v[i] for i in range(len(v))))
    return [ v[i]/vmag for i in range(len(v)) ]

balls, vels, N_pairs = [], [], random.randint(1, 3)
for i in range(N_pairs):
    balls.append([0.5*params['len']+random.uniform(params['rr1'][0], 0)*params['len'], 0.5*params['len']+random.uniform(params['rr1'][0],params['rr1'][1])*params['len'], 0.5*params['len']+random.uniform(params['rr1'][0],params['rr1'][1])*params['len']])
    balls[-1].append(random.uniform(params['rr2'][0], params['rr2'][1])*params['len'])
    vels.append(list(map(operator.sub, [0.5*params['len']]*3, balls[-1][:3])))

    balls.append(list(map(operator.add, [0.5*params['len']]*3, vels[-1])))
    balls[-1].append(random.uniform(params['rr2'][0], params['rr2'][1])*params['len'])
    vels.append(list(map(operator.sub, [0.5*params['len']]*3, balls[-1][:3])))

for i, ball in enumerate(balls):
    ball_c = vec3(ball[0]*scaleToManta+params['bnd'], ball[1]*scaleToManta+params['bnd'], ball[2]*scaleToManta+params['bnd'] if (params['dim']==3) else 0.5)
    ball_i = s.create(Sphere, center=ball_c, radius=ball[3]*scaleToManta)
    begin = pp.size()
    sampleShapeWithParticles(shape=ball_i, flags=gFlags, parts=pp, discretization=params['sres'], randomness=params['jitter'], refillEmpty=True, notiming=True)
    end = pp.size()
    pT.setConstRange(s=FlagFluid, begin=begin, end=end, notiming=True)
    markFluidCells(parts=pp, flags=gFlags, ptype=pT, exclude=FlagObstacle)

    vel = vels[i]
    if (params['dim']<3): vel[2] = 0
    speed = random.uniform(params['rr3'][0], params['rr3'][1])*params['len']
    vel = list(map(operator.mul, normalize(vel), [speed*scaleToManta]*3))
    pV.setConstRange(s=vec3(vel[0], vel[1], vel[2]), begin=begin, end=end, notiming=True)

for i, ball in enumerate(balls):
    ball_c = vec3(ball[0]*scaleToManta+params['bnd'], ball[1]*scaleToManta+params['bnd'], ball[2]*scaleToManta+params['bnd'] if (params['dim']==3) else 0.5)
    ball_i = s.create(Sphere, center=ball_c, radius=ball[3]*scaleToManta)
    gPhi.join(ball_i.computeLevelset(), notiming=True)

if not pargs.nogui:
    gui = Gui()
    gui.show()
    if pargs.pause: gui.pause()

frame_saved = 0
if pargs.output is not None:
    path = '{}/{:05d}/'.format(pargs.output, s.frame)
    os.path.isdir(path) or os.makedirs(path)
    for save, name in savingFuncs:
        save(path+name, notiming=True)

    with open(pargs.output+'/params.pickle', 'wb') as f:
        pickle.dump(params, f)

while (s.timeTotal<params['t_end']): # main loop
    mapPartsToMAC(vel=gV, flags=gFlags, velOld=gVold, parts=pp, partVel=pV, ptype=pT, exclude=FlagOpen|FlagEmpty)

    if params['sdt'] is None: s.adaptTimestep(gV.getMax())
    else: s.adaptTimestepByDt(params['sdt'])

    addGravityNoScale(flags=gFlags, vel=gV, gravity=vec3(0, params['grav'], 0))

    if params['gfm']:
        gridParticleIndex(parts=pp, flags=gFlags, indexSys=gIdxSys, index=gIdx)
        unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=gPhi, radiusFactor=1.0)
        extrapolateLsSimple(phi=gPhi, distance=4, inside=True)
        extrapolateLsSimple(phi=gPhi, distance=4, inside=False)

        if params['stens']:
            getLaplacian(laplacian=gCurv, grid=gPhi)
            gCurv.clamp(-1.0, 1.0)

    setWallBcs(flags=gFlags, vel=gV)
    solvePressure(**paramSolvePressure)
    setWallBcs(flags=gFlags, vel=gV)
    extrapolateMACSimple(flags=gFlags, vel=gV)

    # update velocity (general update from FLIP and individual update for Lagrangian particles)
    flipVelocityUpdate(vel=gV, velOld=gVold, flags=gFlags, parts=pp, partVel=pV, flipRatio=0.97, ptype=pT, exclude=FlagOpen|FlagEmpty)
    pp.updateVelocity(vel=pV, a=vec3(0, params['grav'], 0), dt=s.timestep, ptype=pT, exclude=FlagFluid)

    # update position
    pp.getPosPdata(target=pVtmp)
    pp.advectInGrid(flags=gFlags, vel=gV, integrationMode=IntRK4, deleteInObstacle=False, ptype=pT, exclude=FlagOpen|FlagEmpty)
    pp.advect(vel=pV, ptype=pT, exclude=FlagFluid)
    pp.projectOutOfBnd(flags=gFlags, bnd=params['bnd']+params['dx']*0.5, plane='xXyYzZ', ptype=pT)
    pushOutofObs(parts=pp, flags=gFlags, phiObs=gPhiSld, thresh=params['dx']*0.5, ptype=pT)

    # update velocity of the Lagrangian particles
    pp.updateVelocityFromDeltaPos(vel=pV, x_prev=pVtmp, dt=s.timestep, ptype=pT, exclude=FlagFluid)

    # handling particles going out or coming into the simulation domain
    pp.setType(ptype=pT, mark=FlagOpen, stype=FlagFluid|FlagEmpty, flags=gFlags, cflag=FlagOpen)
    pp.setType(ptype=pT, mark=FlagFluid, stype=FlagOpen, flags=gFlags, cflag=FlagEmpty|FlagFluid)
    markFluidCells(parts=pp, flags=gFlags, ptype=pT, exclude=FlagOpen)
    pp.setType(ptype=pT, mark=FlagFluid, stype=FlagEmpty, flags=gFlags, cflag=FlagFluid)

    if params['dim']==3 and not pargs.nogui:
        gridParticleIndex(parts=pp, flags=gFlags, indexSys=gIdxSys, index=gIdx)
        unionParticleLevelset(parts=pp, indexSys=gIdxSys, flags=gFlags, index=gIdx, phi=gPhi, radiusFactor=1.0)
        extrapolateLsSimple(phi=gPhi, distance=4, inside=True)
        gPhi.createMesh(mesh)

    s.step()

    if (pargs.output is not None) and (frame_saved!=s.frame):
        frame_saved = s.frame
        path = '{}/{:05d}/'.format(pargs.output, s.frame)
        os.path.isdir(path) or os.makedirs(path)
        for save, name in savingFuncs:
            save(path+name, notiming=True)

