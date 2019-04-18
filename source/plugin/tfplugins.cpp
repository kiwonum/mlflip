/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2017 Kiwon Um, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * tensorflor/numpy plugins, mostly for MLFLIP
 * [https://arxiv.org/abs/1704.04456] for now (only compiled if NUMPY is
 * enabled)
 *
 ******************************************************************************/

#include "levelset.h"
#include "commonkernels.h"
#include "particle.h"
#include <cmath>

namespace Manta {

//! simple test kernel and kernel with numpy array
KERNEL(bnd=0)
void knSimpleNumpyTest(Grid<Real>& grid, PyArrayContainer npAr, Real scale)
{
	const float* p = reinterpret_cast<float*>(npAr.pData);
	grid(i,j,k) += scale * (Real)p[j*grid.getSizeX()+i]; // calc access into numpy array, no size check here!
}

//! simple test function and kernel with numpy array
PYTHON()
void simpleNumpyTest( Grid<Real>& grid, PyArrayContainer npAr, Real scale) {
	knSimpleNumpyTest(grid, npAr, scale);
}

//! extract feature vectors

KERNEL(pts)
void knExtractFeatureVel(
	const BasicParticleSystem &p, Real *fv, const IndexInt N_row, const IndexInt off_begin,
	const MACGrid &vel, const Real scale, const ParticleDataImpl<int> *ptype, const int exclude,
	const int window, const Real h) {
	if(!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;

	const int _k = (vel.is3D()) ? -window : 0, K = (vel.is3D()) ? window : 0;
	const IndexInt D = (vel.is3D()) ? 3 : 2;
	const IndexInt off_idx = idx*N_row;

	IndexInt off_stencil = 0;
	for(int i=-window; i<=window; ++i) {
		for(int j=-window; j<=window; ++j) {
			for(int k=_k; k<=K; ++k) {
				const Vec3 off_pos(static_cast<Real>(i)*h, static_cast<Real>(j)*h, static_cast<Real>(k)*h);
				const Vec3 pos_s = p[idx].pos + off_pos;

				const Vec3 vel_s = vel.getInterpolated(pos_s)*scale;
				const IndexInt off_vel = off_idx + off_begin + off_stencil*D;
				fv[off_vel + 0] = vel_s[0];
				fv[off_vel + 1] = vel_s[1];
				if(vel.is3D()) fv[off_vel + 2] = vel_s[2];

				++off_stencil;
			}
		}
	}
}
KERNEL(pts)
void knExtractFeatureVelRel(
	const BasicParticleSystem &p, Real *fv, const IndexInt N_row, const IndexInt off_begin,
	const MACGrid &vel, const Real scale, const ParticleDataImpl<int> *ptype, const int exclude,
	const int window, const Real h) {
	if(!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;

	const int _k = (vel.is3D()) ? -window : 0, K = (vel.is3D()) ? window : 0;
	const IndexInt D = (vel.is3D()) ? 3 : 2;
	const IndexInt off_idx = idx*N_row;
	const Vec3 v0 = vel.getInterpolated(p[idx].pos);

	IndexInt off_stencil = 0;
	for(int i=-window; i<=window; ++i) {
		for(int j=-window; j<=window; ++j) {
			for(int k=_k; k<=K; ++k) {
				const Vec3 off_pos(static_cast<Real>(i)*h, static_cast<Real>(j)*h, static_cast<Real>(k)*h);
				const Vec3 pos_s = p[idx].pos + off_pos;

				const Vec3 vel_s = (vel.getInterpolated(pos_s) - v0)*scale;
				const IndexInt off_vel = off_idx + off_begin + off_stencil*D;
				fv[off_vel + 0] = vel_s[0];
				fv[off_vel + 1] = vel_s[1];
				if(vel.is3D()) fv[off_vel + 2] = vel_s[2];

				if(i==0 && j==0 && k==0) {
					fv[off_vel + 0] = scale;
					fv[off_vel + 1] = scale;
					if(vel.is3D()) fv[off_vel + 2] = scale;
				}

				++off_stencil;
			}
		}
	}
}

KERNEL(pts)
void knExtractFeaturePhi(
	const BasicParticleSystem &p, Real *fv, const IndexInt N_row, const IndexInt off_begin,
	const Grid<Real> &phi, const Real scale, const ParticleDataImpl<int> *ptype, const int exclude,
	const int window, const Real h) {
	if(!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;

	const int _k = (phi.is3D()) ? -window : 0, K = (phi.is3D()) ? window : 0;
	const IndexInt off_idx = idx*N_row;

	IndexInt off_stencil = 0;
	for(int i=-window; i<=window; ++i) {
		for(int j=-window; j<=window; ++j) {
			for(int k=_k; k<=K; ++k) {
				const Vec3 off_pos(static_cast<Real>(i)*h, static_cast<Real>(j)*h, static_cast<Real>(k)*h);
				const Vec3 pos_s = p[idx].pos + off_pos;

				const Real phi_s = phi.getInterpolated(pos_s)*scale;
				const IndexInt off_phi = off_idx + off_begin + off_stencil;
				fv[off_phi] = phi_s;

				++off_stencil;
			}
		}
	}
}

KERNEL(pts)
void knExtractFeatureGeo(
	const BasicParticleSystem &p, Real *fv, const IndexInt N_row, const IndexInt off_begin,
	const FlagGrid &geo, const Real scale, const ParticleDataImpl<int> *ptype, const int exclude,
	const int window, const Real h) {
	if(!p.isActive(idx) || (ptype && ((*ptype)[idx] & exclude))) return;

	const int _k = (geo.is3D()) ? -window : 0, K = (geo.is3D()) ? window : 0;
	const IndexInt off_idx = idx*N_row;

	IndexInt off_stencil = 0;
	for(int i=-window; i<=window; ++i) {
		for(int j=-window; j<=window; ++j) {
			for(int k=_k; k<=K; ++k) {
				const Vec3 off_pos(static_cast<Real>(i)*h, static_cast<Real>(j)*h, static_cast<Real>(k)*h);
				const Vec3 pos_s = p[idx].pos + off_pos;

				const Real geo_s = static_cast<Real>(geo.getAt(pos_s))*scale;
				const IndexInt off_geo = off_idx + off_begin + off_stencil;
				fv[off_geo] = geo_s;

				++off_stencil;
			}
		}
	}
}

PYTHON()
void extractFeatureVel(
	PyArrayContainer fv, const int N_row, const int off_begin,
	const BasicParticleSystem &p, const MACGrid &vel,
	const Real scale=1.0, const ParticleDataImpl<int> *ptype=NULL, const int exclude=0,
	const int window=1, const Real h=1.0) {
	knExtractFeatureVel(p, reinterpret_cast<Real*>(fv.pData), N_row, off_begin, vel, scale, ptype, exclude, window, h);
}
PYTHON()
void extractFeatureVelRel(
	PyArrayContainer fv, const int N_row, const int off_begin,
	const BasicParticleSystem &p, const MACGrid &vel,
	const Real scale=1.0, const ParticleDataImpl<int> *ptype=NULL, const int exclude=0,
	const int window=1, const Real h=1.0) {
	knExtractFeatureVelRel(p, reinterpret_cast<Real*>(fv.pData), N_row, off_begin, vel, scale, ptype, exclude, window, h);
}
PYTHON()
void extractFeaturePhi(
	PyArrayContainer fv, const int N_row, const int off_begin,
	const BasicParticleSystem &p, const Grid<Real> &phi,
	const Real scale=1.0, const ParticleDataImpl<int> *ptype=NULL, const int exclude=0,
	const int window=1, const Real h=1.0) {
	knExtractFeaturePhi(p, reinterpret_cast<Real*>(fv.pData), N_row, off_begin, phi, scale, ptype, exclude, window, h);
}
PYTHON()
void extractFeatureGeo(
	PyArrayContainer fv, const int N_row, const int off_begin,
	const BasicParticleSystem &p, const FlagGrid &flag,
	const Real scale=1.0, const ParticleDataImpl<int> *ptype=NULL, const int exclude=0,
	const int window=1, const Real h=1.0) {
	knExtractFeatureGeo(p, reinterpret_cast<Real*>(fv.pData), N_row, off_begin, flag, scale, ptype, exclude, window, h);
}

// non-numpy related helpers

//! region detection functions

PYTHON()
void extendRegion(FlagGrid &flags, const int region, const int exclude, const int depth) {
	flags.extendRegion(region, exclude, depth);
}

// particle helpers

// TODO: merge with KnUpdateVelocity (particle.h); duplicated
KERNEL(pts)
void KnAddForcePvel(ParticleDataImpl<Vec3> &v, const Vec3 &da, const ParticleDataImpl<int> *ptype, const int exclude) {
	if(ptype && ((*ptype)[idx] & exclude)) return;
	v[idx] += da;
}
//! add force to vec3 particle data (ie, a velocity)
// TODO: merge with ParticleSystem::updateVelocity (particle.h); duplicated
PYTHON()
void addForcePvel(ParticleDataImpl<Vec3> &vel, const Vec3 &a, const Real dt, const ParticleDataImpl<int> *ptype, const int exclude) {
	KnAddForcePvel(vel, a*dt, ptype, exclude);
}

//! retrieve velocity from position change
PYTHON()
void updateVelocityFromDeltaPos(BasicParticleSystem& parts, ParticleDataImpl<Vec3> &vel, const ParticleDataImpl<Vec3> &x_prev, const Real dt, const ParticleDataImpl<int> *ptype, const int exclude) {
	parts.updateVelocityFromDeltaPos(vel, x_prev, dt, ptype, exclude);
}

//! simple foward Euler integration for particle system
PYTHON()
void eulerStep(BasicParticleSystem& parts, const ParticleDataImpl<Vec3> &vel, const ParticleDataImpl<int> *ptype, const int exclude) {
	parts.advect(vel, ptype, exclude);
}

// TODO: merge with KnSetType (particle.h); duplicated
KERNEL(pts)
void KnSetPartType(ParticleDataImpl<int> &ptype, BasicParticleSystem &part, const int mark, const int stype, const FlagGrid &flags, const int cflag) {
	if(flags.isInBounds(part.getPos(idx), 0) && (flags.getAt(part.getPos(idx))&cflag) && (ptype[idx]&stype)) ptype[idx] = mark;
}
// TODO: merge with ParticleSystem::setType (particle.h); duplicated
PYTHON()
void setPartType(BasicParticleSystem &parts, ParticleDataImpl<int> &ptype, const int mark, const int stype, const FlagGrid &flags, const int cflag) {
	KnSetPartType(ptype, parts, mark, stype, flags, cflag);
}

} //namespace
