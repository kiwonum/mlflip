/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2015-2016 Kiwon Um, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL)
 * http://www.gnu.org/licenses
 *
 * SPH (smoothed particle hydrodynamics) plugins
 *
 * TODO: Different kernels should be usable.
 *
 ******************************************************************************/

#include "matrixbase.h"
#include "particle.h"
#include "pneighbors.h"
#include "sphkernel.h"
#include "sphsolver.h"

namespace Manta {

KERNEL(pts)
void knSphUpdateVelocity(
	ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Vec3> &vn, const ParticleDataImpl<Vec3> &f, const Real dt,
	const SphWorld &sph, const int itype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;
	v[idx] = vn[idx] + f[idx]*dt/sph._m0;
}
PYTHON()
void sphUpdateVelocity(
	ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Vec3> &vn, const ParticleDataImpl<Vec3> &f, const Real dt,
	const SphWorld &sph,
	const int itype=SphWorld::FlagFluid) {
	knSphUpdateVelocity(v, vn, f, dt, sph, itype);
}
KERNEL(pts)
void knSphSetBndVelocity(
	ParticleDataImpl<Vec3> &v,
	const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;
	v[idx] = Vec3(0.0);

	Real sum_w = 0.0;
	Vec3 sum_vw(0.0);
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it); if(!(t[j]&jtype)) continue;
		const Real w = k.f(n.length(it));
		sum_w += w;
		sum_vw += v[j]*w;
	}
	if(sum_w>Real(0)) v[idx] = -sum_vw/sum_w;
}
PYTHON()
void sphSetBndVelocity(
	ParticleDataImpl<Vec3> &v, const CubicSpline &k,
	const SphWorld &sph,
	const int itype=SphWorld::FlagObstacle,
	const int jtype=SphWorld::FlagFluid) {
	knSphSetBndVelocity(v, k, sph, itype, jtype);
}
KERNEL(pts)
void knSphUpdatePosition(
	BasicParticleSystem &x, const ParticleDataImpl<Vec3> &v, const Real dt,
	const SphWorld &sph, const int itype) {
	const ParticleDataImpl<int> &t = sph.particleType();

	if(!x.isActive(idx) || !(t[idx]&itype)) return;
	x[idx].pos += dt*v[idx];
}
PYTHON()
void sphUpdatePosition(
	BasicParticleSystem &x, const ParticleDataImpl<Vec3> &v, const Real dt,
	const SphWorld &sph,
	const int itype=SphWorld::FlagFluid) {
	knSphUpdatePosition(x, v, dt, sph, itype);
}
KERNEL(pts)
void knSphComputeDensity(
	ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> *m,
	const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;
	Real sum_w = 0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it); if(!(t[j]&jtype)) continue;
		const Real len_xij = n.length(it);
		sum_w += ((m) ? m->get(j) : sph._m0) * k.f(len_xij);
	}
	d[idx] = sum_w;
}
PYTHON()
void sphComputeDensity(
	ParticleDataImpl<Real> &d, const CubicSpline &k,
	const SphWorld &sph, ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputeDensity(d, m, k, sph, itype, jtype);
}
KERNEL(pts)
void knSphComputeVolume(
	ParticleDataImpl<Real> &d, const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;
	Real sum_w = 0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it); if(!(t[j]&jtype)) continue;
		const Real len_xij = n.length(it);
		sum_w += k.f(len_xij);
	}
	d[idx] = 1.0/sum_w;
}
PYTHON()
void sphComputeVolume(
	ParticleDataImpl<Real> &d, const CubicSpline &k, const SphWorld &sph,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid) {
	knSphComputeVolume(d, k, sph, itype, jtype);
}
KERNEL(pts)
void knSphComputeDivergence(
	ParticleDataImpl<Real> &div, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Vec3> &v,
	const ParticleDataImpl<Real> *m, const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos, &vi = v[idx];
	Real sum_div = 0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos, &vj = v[j];
		Vec3 xij = xi-xj, vij = vi-vj; if(pts.getParent()->is2D()) xij.z = vij.z = 0.0;
		const Real Vj = ((m) ? m->get(j) : sph._m0)/d[j];
		sum_div += dot(vij, k.grad_w(xij, len_xij))*Vj;
	}
	div[idx] = -sum_div;
}
PYTHON()
void sphComputeDivergence(
	ParticleDataImpl<Real> &div, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Vec3> &v,
	const CubicSpline &k, const SphWorld &sph, const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputeDivergence(div, d, v, m, k, sph, itype, jtype);
}
KERNEL(pts)
void knSphComputeDivergenceSimple(
	ParticleDataImpl<Real> &d, const ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Real> *m,
	const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos, &vi = v[idx];
	Real sum_div = 0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos, &vj = v[j];
		Vec3 xij = xi - xj, vij = vi - vj; if(pts.getParent()->is2D()) xij.z = vij.z = 0.0;
		sum_div += dot(vij, k.grad_w(xij, len_xij))*((m) ? m->get(j) : sph._m0);
	}
	d[idx] = -sum_div;
}
PYTHON()
void sphComputeDivergenceSimple(
	ParticleDataImpl<Real> &div, const ParticleDataImpl<Vec3> &v,
	const CubicSpline &k, const SphWorld &sph,
	const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputeDivergenceSimple(div, v, m, k, sph, itype, jtype);
}
KERNEL(pts)
void knSphReinitDensity(
	ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> &dn,
	const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	Real sum_w = 0.0, sum_w_over_d = 0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it); if(!(t[j]&jtype)) continue;
		const Real len_xij = n.length(it);
		const Real w = k.f(len_xij);
		sum_w += w;
		sum_w_over_d += w/dn[j];
	}
	d[idx] = sum_w/sum_w_over_d; // NOTE: always sum_w_over_d > 0
}
PYTHON()
void sphReinitDensity(
	ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> &dn,
	const CubicSpline &k, const SphWorld &sph,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphReinitDensity(d, dn, k, sph, itype, jtype);
}
KERNEL(pts)
void knSphComputePressure(
	ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d,
	const SphWorld &sph, const int itype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;
	p[idx] = sph.equationOfState(d[idx], sph._d0, sph._p0, sph._gamma);
}
PYTHON()
void sphComputePressure(
	ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d,
	const SphWorld &sph,
	const int itype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputePressure(p, d, sph, itype);
}
KERNEL(pts)
void knSphComputeCurl(
	ParticleDataImpl<Vec3> &w, const ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Real> &d,
	const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos, &vi = v[idx];
	Vec3 sum_v(0.0);
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);
		const Real len_xij = n.length(it);
		if(!(t[j]&jtype) || idx==j || len_xij<=Real(0)) continue;

		const Vec3 &xj = pts[j].pos, &vj = v[j];
		Vec3 xij = xi - xj;
		Vec3 vij = vi - vj; if(pts.getParent()->is2D()) xij.z = vij.z = 0.0;
		const Real Vj = sph._m0/d[j];
		sum_v += cross(vij, k.grad_w(xij, len_xij))*Vj;
	}
	w[idx] = sum_v;
}
PYTHON()
void sphComputeCurl(
	ParticleDataImpl<Vec3> &w, const ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Real> &d, const CubicSpline &k,
	const SphWorld &sph,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputeCurl(w, v, d, k, sph, itype, jtype);
}
KERNEL(pts)
void knSphAddJacobian(
	ParticleDataImpl<Vec3> &diag, ParticleDataImpl<Vec3> &offu, ParticleDataImpl<Vec3> &offl,
	const ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Real> &d,
	const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos, &vi = v[idx];
	Vec3 sum_diag(0.0), sum_offu(0.0), sum_offl(0.0);
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);
		const Real len_xij = n.length(it);
		if(!(t[j]&jtype) || idx==j || len_xij<=Real(0)) continue;

		const Vec3 &xj = pts[j].pos, &vj = v[j];
		Vec3 xij = xi - xj;
		Vec3 vji = vj - vi; if(pts.getParent()->is2D()) xij.z = vji.z = 0.0;
		const Vec3 grad_w = k.grad_w(xij, len_xij);
		const Real Vj = sph._m0/d[j];
		sum_diag += vji*grad_w*Vj;
		sum_offu.x += vji.x*grad_w.y*Vj; // xy
		sum_offu.y += vji.y*grad_w.z*Vj; // yz
		sum_offu.z += vji.z*grad_w.x*Vj; // zx
		sum_offl.x += vji.y*grad_w.x*Vj; // yx
		sum_offl.y += vji.z*grad_w.y*Vj; // zy
		sum_offl.z += vji.x*grad_w.z*Vj; // xz
	}
	diag[idx] += sum_diag;
	offu[idx] += sum_offu;
	offl[idx] += sum_offl;
}
//	11 12 13
// S =	21 22 23
//	31 32 33
// diag=(11 22 33), offu = (12, 23, 13), offl = (21, 32, 31)
PYTHON()
void sphAddJacobian(
	ParticleDataImpl<Vec3> &diag, ParticleDataImpl<Vec3> &offu, ParticleDataImpl<Vec3> &offl,
	const ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Real> &d, const CubicSpline &k,
	const SphWorld &sph,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphAddJacobian(diag, offu, offl, v, d, k, sph, itype, jtype);
}
KERNEL(pts)
void knSphSetBndValues(
	ParticleDataImpl<Real> &d, ParticleDataImpl<Real> &p, ParticleDataImpl<Real> &Vsqr,
	const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype, const bool clamp) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;
	const Vec3 &xi = pts[idx].pos;

	d[idx]	  = sph._d0;
	p[idx]	  = 0.0;
	Vsqr[idx] = square(sph._m0/sph._d0);

	Real sum_w = 0.0, sum_pw=0.0;
	Vec3 sum_dx_wf(0.0);
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);
		if(!(t[j]&jtype)) continue;
		const Real w = k.f(n.length(it));
		const Vec3 &xj = pts[j].pos;
		sum_w += w;
		sum_pw += p[j]*w;
		sum_dx_wf += d[j]*(xi -xj)*w;
	}
	if(sum_w>Real(0)) {
		p[idx]	  = (sum_pw + dot(sph._g, sum_dx_wf))/sum_w;
		if(clamp) p[idx] = std::max(Real(0), p[idx]);
		d[idx]	  = sph._d0*std::pow(p[idx]/sph._p0 + Real(1), Real(1)/sph._gamma);
		Vsqr[idx] = square(sph._m0/d[idx]);
	}
}
PYTHON()
void sphSetBndValues(
	ParticleDataImpl<Real> &d, ParticleDataImpl<Real> &p, ParticleDataImpl<Real> &Vsqr,
	const CubicSpline &k, const SphWorld &sph,
	const int itype=SphWorld::FlagObstacle,
	const int jtype=SphWorld::FlagFluid,
	const bool clamp=false) {
	knSphSetBndValues(d, p, Vsqr, k, sph, itype, jtype, clamp);
}
KERNEL(pts)
void knSphComputeIisphDii(
	ParticleDataImpl<Vec3> &dii, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> *m,
	const CubicSpline &k, const SphWorld &sph, const Real dt, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos;
	const Real overdi2 = safeDivide(Real(1.0), square(d[idx]));

	dii[idx].x = dii[idx].y = dii[idx].z = 0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos;
		dii[idx] -= k.grad_w(xi - xj, len_xij)*((m) ? m->get(j) : sph._m0)*overdi2;
	}
}
PYTHON()
void sphComputeIisphDii(
	ParticleDataImpl<Vec3> &dii, const ParticleDataImpl<Real> &d,
	const CubicSpline &k, const SphWorld &sph, const Real dt, const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputeIisphDii(dii, d, m, k, sph, dt, itype, jtype);
}
KERNEL(pts)
void knSphComputeIisphAii(
	ParticleDataImpl<Real> &aii, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Vec3> &dii,
	const ParticleDataImpl<Real> *m, const CubicSpline &k, const SphWorld &sph, const Real dt,
	const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos;
	const Real mi_di2 = safeDivide((m) ? m->get(idx) : sph._m0, square(d[idx]));

	aii[idx] = 0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos;
		const Vec3 grad_w = k.grad_w(xi - xj, len_xij);
		const Vec3 dji = (t[j]&SphWorld::FlagFluid) ? grad_w*mi_di2 : Vec3(0.0);
		aii[idx] += dot(dii[idx] - dji, grad_w)*((m) ? m->get(j) : sph._m0);
	}
}
PYTHON()
void sphComputeIisphAii(
	ParticleDataImpl<Real> &aii, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Vec3> &dii,
	const CubicSpline &k, const SphWorld &sph, const Real dt, const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputeIisphAii(aii, d, dii, m, k, sph, dt, itype, jtype);
}
KERNEL(pts)
void knSphComputeIisphDijPj(
	ParticleDataImpl<Vec3> &dijpj, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> &p,
	const ParticleDataImpl<Real> *m, const CubicSpline &k, const SphWorld &sph, const Real dt,
	const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos;

	dijpj[idx].x = dijpj[idx].y = dijpj[idx].z = 0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos;
		dijpj[idx] -= safeDivide(k.grad_w(xi - xj, len_xij)*((m) ? m->get(j) : sph._m0)*p[j], Vec3(square(d[j])));
	}
}
PYTHON()
void sphComputeIisphDijPj(
	ParticleDataImpl<Vec3> &dijpj, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> &p,
	const CubicSpline &k, const SphWorld &sph, const Real dt, const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputeIisphDijPj(dijpj, d, p, m, k, sph, dt, itype, jtype);
}
KERNEL(pts)
void knSphComputeIisphP(
	ParticleDataImpl<Real> &p_next, const ParticleDataImpl<Real> &p,
	const ParticleDataImpl<Real> &d_adv, const ParticleDataImpl<Real> &d,
	const ParticleDataImpl<Real> &aii, const ParticleDataImpl<Vec3> &dii, const ParticleDataImpl<Vec3> &dijpj,
	const ParticleDataImpl<Real> *m, const CubicSpline &k, const SphWorld &sph, const Real dt,
	const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;
	if(aii[idx]==Real(0)) return; // isolated particle

	const Vec3 &xi = pts[idx].pos;
	const Real overdt2 = safeDivide(Real(1.0), square(dt));
	const Real mi_di2 = safeDivide((m) ? m->get(idx) : sph._m0, square(d[idx]));

	Real sumv=0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos;
		const Vec3 grad_w = k.grad_w(xi-xj, len_xij);
		const Vec3 dji = grad_w*mi_di2;
		const Real mj = ((m) ? m->get(j) : sph._m0);
		const Vec3 dFi = dijpj[idx];
		const Vec3 dFj = (t[j]&SphWorld::FlagFluid) ? dii[j]*p[j] + dijpj[j] - dji*p[idx] : Vec3(0.0);
		sumv += dot(dFi - dFj, grad_w)*mj;
	}
	p_next[idx] = 0.5*p[idx] + 0.5*safeDivide((sph._d0 - d_adv[idx])*overdt2 - sumv, aii[idx]);
}
PYTHON()
void sphComputeIisphP(
	ParticleDataImpl<Real> &p_next, const ParticleDataImpl<Real> &p,
	const ParticleDataImpl<Real> &d_adv, const ParticleDataImpl<Real> &d,
	const ParticleDataImpl<Real> &aii, const ParticleDataImpl<Vec3> &dii, const ParticleDataImpl<Vec3> &dijpj,
	const CubicSpline &k, const SphWorld &sph, const Real dt, const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputeIisphP(p_next, p, d_adv, d, aii, dii, dijpj, m, k, sph, dt, itype, jtype);
}
KERNEL(pts)
void knSphComputeIisphD(
	ParticleDataImpl<Real> &d_next, const ParticleDataImpl<Real> &d_adv, const ParticleDataImpl<Real> &d,
	const ParticleDataImpl<Real> &p, const ParticleDataImpl<Vec3> &dii, const ParticleDataImpl<Vec3> &dijpj,
	const ParticleDataImpl<Real> *m, const CubicSpline &k, const SphWorld &sph, const Real dt,
	const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos;

	Real sumv=0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos;
		const Vec3 grad_w = k.grad_w(xi-xj, len_xij);
		const Real mj = ((m) ? m->get(j) : sph._m0);
		const Vec3 dFi = (t[j]&SphWorld::FlagFluid) ? dii[idx]*p[idx] + dijpj[idx] : -safeDivide(mj, square(d[idx]))*p[idx]*grad_w;
		const Vec3 dFj = (t[j]&SphWorld::FlagFluid) ? dii[j]*p[j] + dijpj[j] : Vec3(0.0);
		sumv += dot(dFi - dFj, grad_w)*mj;
	}
	d_next[idx] = d_adv[idx] + square(dt)*sumv;
}
PYTHON()
void sphComputeIisphD(
	ParticleDataImpl<Real> &d_next, const ParticleDataImpl<Real> &d_adv, const ParticleDataImpl<Real> &d,
	const ParticleDataImpl<Real> &p, const ParticleDataImpl<Vec3> &dii, const ParticleDataImpl<Vec3> &dijpj,
	const CubicSpline &k, const SphWorld &sph, const Real dt, const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputeIisphD(d_next, d_adv, d, p, dii, dijpj, m, k, sph, dt, itype, jtype);
}
PYTHON()
Real sphComputePcisphDelta(const CubicSpline &k, const SphWorld &sph) {
	const int r  = static_cast<int>(k.supportRadius()/sph._delta) + 1;
	int	  k0 = -r, k1 = r;
	if(sph.particleSystem().getParent()->is2D()) { k0 = 0; k1 = 1; }
	const Real Rsqr = square(k.supportRadius());
	const Real over_beta = 0.5*square(sph._d0/sph._m0);

	Vec3 sum_gw(0.0);
	Real sum_gwgw=0.0;
	for(int di=-r; di<r; ++di) {
		for(int dj=-r; dj<r; ++dj) {
			for(int dk=k0; dk<k1; ++dk) {
				const Vec3 xj(sph._delta*static_cast<Real>(di), sph._delta*static_cast<Real>(dj), sph._delta*static_cast<Real>(dk));
				const Vec3 xij = -xj;
				const Real lensqr_xij = normSquare(xij);
				if(lensqr_xij<=Real(0) || Rsqr<lensqr_xij) continue;

				const Real len_xij = std::sqrt(lensqr_xij);
				const Vec3 grad_w = k.derivative_f(len_xij)*xij/len_xij;
				sum_gw += grad_w;
				sum_gwgw += dot(grad_w, grad_w);
			}
		}
	}

	return over_beta/(dot(sum_gw, sum_gw) + sum_gwgw);
}
KERNEL(pts)
void knSphComputeDfsphAlpha(
	ParticleDataImpl<Real> &a, const ParticleDataImpl<Real> &d,
	const ParticleDataImpl<Real> *m, const CubicSpline &k, const SphWorld &sph,
	const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos;
	Vec3 sum_mgw(0.0);
	Real sum_mgw_sqr = 0.0;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos;
		const Vec3 gw_ij = k.grad_w(xi - xj, len_xij);
		const Real mj = ((m) ? m->get(j) : sph._m0);
		sum_mgw += mj*gw_ij;
		sum_mgw_sqr += (t[j]&SphWorld::FlagFluid) ? square(mj)*normSquare(gw_ij) : 0.0;
	}
	a[idx] = d[idx]/std::max(VECTOR_EPSILON, normSquare(sum_mgw) + sum_mgw_sqr);
}
PYTHON()
void sphComputeDfsphAlpha(
	ParticleDataImpl<Real> &a, const ParticleDataImpl<Real> &d,
	const CubicSpline &k, const SphWorld &sph, const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphComputeDfsphAlpha(a, d, m, k, sph, itype, jtype);
}
KERNEL(pts)
void knSphStompIsolatedParticleValue(
	ParticleDataImpl<Real> &v, const SphWorld &sph,	const int itype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;
	if(n.size(idx)<=1) v[idx] = 0.0;
}
PYTHON()
void sphStompIsolatedParticleValue(ParticleDataImpl<Real> &v, const SphWorld &sph, const int itype=SphWorld::FlagFluid) {
	knSphStompIsolatedParticleValue(v, sph, itype);
}

KERNEL(pts)
void knSphComputeConstantForce(
	ParticleDataImpl<Vec3> &f, const Vec3 &v,
	const SphWorld &sph, const int itype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;
	f[idx] += v;
}
PYTHON()
void sphComputeConstantForce(
	ParticleDataImpl<Vec3> &f, const Vec3 &v,
	const SphWorld &sph,
	const int itype=SphWorld::FlagFluid,
	const bool accumulate=true) {
	if(!accumulate) f.setConst(Vec3(0.0));
	knSphComputeConstantForce(f, v, sph, itype);
}
KERNEL(pts) template<typename T_Pij>
void knSphComputePressureForce(
	ParticleDataImpl<Vec3> &f, const T_Pij &Pij,
	const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos;
	Vec3 fi(0.0);
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos;
		fi -= k.grad_w(xi - xj, len_xij)*Pij(idx, j);
	}
	f[idx] += fi;
}
PYTHON()
void sphComputePressureForce(
	ParticleDataImpl<Vec3> &f, const ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d,
	const CubicSpline &k, const SphWorld &sph, const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle,
	const bool accumulate=true) {
	if(!accumulate) f.setConst(Vec3(0.0));
	if(m) knSphComputePressureForce<PijMij>(f, PijMij(p, d, *m, sph.particleType()), k, sph, itype, jtype);
	else knSphComputePressureForce<Pij>(f, Pij(p, d, sph._m0, sph.particleType()), k, sph, itype, jtype);
}
PYTHON()
void sphComputeDfsphPressureForce(
	ParticleDataImpl<Vec3> &f, const ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d,
	const CubicSpline &k, const SphWorld &sph, const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle,
	const bool accumulate=true) {
	if(!accumulate) f.setConst(Vec3(0.0));
	if(m) knSphComputePressureForce<PijMijDfsph>(f, PijMijDfsph(p, d, *m, sph.particleType()), k, sph, itype, jtype);
	else knSphComputePressureForce<PijDfsph>(f, PijDfsph(p, d, sph._m0, sph.particleType()), k, sph, itype, jtype);
}
PYTHON()
void sphComputeDensityWeightedPressureForce(
	ParticleDataImpl<Vec3> &f, const ParticleDataImpl<Real> &p,
	const ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> &Vsqr,
	const CubicSpline &k, const SphWorld &sph,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle,
	const bool accumulate=true) {
	if(!accumulate) f.setConst(Vec3(0.0));
	knSphComputePressureForce<PijWeighted>(f, PijWeighted(p, d, Vsqr), k, sph, itype, jtype);
}
PYTHON()
void sphComputeBackgroundPressureForce(
	ParticleDataImpl<Vec3> &f, const ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &Vsqr,
	const CubicSpline &k, const SphWorld &sph,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle,
	const bool accumulate=true) {
	if(!accumulate) f.setConst(Vec3(0.0));
	knSphComputePressureForce<PijBg>(f, PijBg(p, Vsqr), k, sph, itype, jtype);
}
KERNEL(pts)
void knSphComputeBoundaryForce(
	ParticleDataImpl<Vec3> &f,
	const BndKernel &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos;
	Vec3 fi(0.0);
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(k.supportRadius()<len_xij) continue;
		const Vec3 &xj = pts[j].pos;
		fi += (xi - xj)*k.f(len_xij)/len_xij; // NOTE: [Monaghan et al. 2005] used slightly different one.
	}
	f[idx] += fi*0.5;
}
PYTHON()
void sphComputeBoundaryForce(
	ParticleDataImpl<Vec3> &f,
	const BndKernel &k, const SphWorld &sph,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagObstacle,
	const bool accumulate=true) {
	if(!accumulate) f.setConst(Vec3(0.0));
	knSphComputeBoundaryForce(f, k, sph, itype, jtype);
}
KERNEL(pts)
void knSphSetBndPressure(
	ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d,
	const Real pb, const Real d_th,
	const SphWorld &sph, const int itype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;
	if(d[idx]<d_th) p[idx] = pb;
}
PYTHON()
void sphSetBndPressure(
	ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d,
	const Real pb, const Real d_th,
	const SphWorld &sph,
	const int itype=SphWorld::FlagFluid) {
	knSphSetBndPressure(p, d, pb, d_th, sph, itype);
}
KERNEL(pts)
void knSphComputeArtificialViscousForce(
	ParticleDataImpl<Vec3> &f, const ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> *m,
	const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos, &vi = v[idx];
	const Real di = d[idx], mi = (m) ? m->get(idx) : sph._m0;
	Vec3 fi(0.0);
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos, &vj = v[j];
		const Vec3 xij = xi - xj, vij = vi - vj;
		const Real v_dot_x = dot(vij, xij) + ((pts.getParent()->is3D()) ? 0.0 : -vij.z*xij.z);
		if(v_dot_x<Real(0)) {
			const Real dj = ((t[j]&SphWorld::FlagObstacle) ? di : d[j]), mj = (m) ? m->get(j) : sph._m0;
			const Real nu = 2.0*sph._alpha*k.radius()*sph._c/(di+dj);
			fi += k.grad_w(xi - xj, len_xij)*mi*mj*nu*v_dot_x/(square(len_xij) + 0.01*square(k.radius()));
		}
	}
	f[idx] += fi;
}
PYTHON()
void sphComputeArtificialViscousForce(
	ParticleDataImpl<Vec3> &f, const ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Real> &d,
	const CubicSpline &k, const SphWorld &sph, const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle,
	const bool accumulate=true) {
	if(!accumulate) f.setConst(Vec3(0.0));
	knSphComputeArtificialViscousForce(f, v, d, m, k, sph, itype, jtype);
}
// [Becker and Teschner 2007]
KERNEL(pts)
void knSphComputeSurfTension(
	ParticleDataImpl<Vec3> &f, const ParticleDataImpl<Real> *m,
	const CubicSpline &k, const SphWorld &sph, const Real kappa,
	const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos;
	Vec3 sum_f(0.0);
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos;
		const Vec3 xij = xi - xj;
		sum_f += xij*((m) ? m->get(j) : sph._m0)*k.f(len_xij);
	}
	f[idx] -= sum_f*kappa;
}
PYTHON()
void sphComputeSurfTension(
	ParticleDataImpl<Vec3> &f, const CubicSpline &k, const SphWorld &sph, const Real kappa,
	const ParticleDataImpl<Real> *m=NULL,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle,
	const bool accumulate=true) {
	if(!accumulate) f.setConst(Vec3(0.0));
	knSphComputeSurfTension(f, m, k, sph, kappa, itype, jtype);
}

KERNEL(pts)
void knSphComputeTensorForce(
	ParticleDataImpl<Vec3> &f, const ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Vec3> &vv,
	const ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> &Vsqr,
	const CubicSpline &k, const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos;
	const Real Vsqr_i = Vsqr[idx];
	const Matrix3x3f A_i = outerProduct(v[idx], vv[idx]-v[idx])*d[idx];

	Vec3 fi(0.0);
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j|| !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;

		const Vec3 &xj = pts[j].pos;
		const Real deriv_w = k.derivative_f(len_xij);
		const Real Vsqr_j = Vsqr[j];
		Vec3 sum_Vsqr_g_w = (xi - xj)*(deriv_w*(Vsqr_i+Vsqr_j)/len_xij);
		if(pts.getParent()->is2D()) sum_Vsqr_g_w.z = 0.0;

		const Matrix3x3f A_j = outerProduct(v[j], vv[j]-v[j])*d[j];

		fi += 0.5*(A_i+A_j)*sum_Vsqr_g_w;
	}
	if(pts.getParent()->is2D()) fi.z = 0.0;
	f[idx] += fi;
}
PYTHON()
void sphComputeTensorForce(
	ParticleDataImpl<Vec3> &f, const ParticleDataImpl<Vec3> &v, const ParticleDataImpl<Vec3> &vv,
	const ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> &Vsqr,
	const CubicSpline &k, const SphWorld &sph,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle,
	const bool accumulate=true) {
	if(!accumulate) f.setConst(Vec3(0.0));
	knSphComputeTensorForce(f, v, vv, d, Vsqr, k, sph, itype, jtype);
}

KERNEL(pts)
void knSphAddPositionCorrection(
	ParticleDataImpl<Vec3> &dp, const BasicParticleSystem &x,
	const SphWorld &sph, const int itype, const int jtype) {
	const BasicParticleSystem	&pts = sph.particleSystem();
	const ParticleDataImpl<int>	&t   = sph.particleType();
	const ParticleNeighbors		&n   = sph.neighborData();

	if(!pts.isActive(idx) || !(t[idx]&itype)) return;

	const Vec3 &xi = pts[idx].pos;
	for(ParticleNeighbors::Neighbors::const_iterator it=n.begin(idx); it!=n.end(idx); ++it) {
		const int j = n.neighborIdx(it);   if(idx==j || !(t[j]&jtype)) continue;
		const Real len_xij = n.length(it); if(len_xij<=Real(0)) continue;
		const Vec3 &xj = pts[j].pos;
		const Vec3 xij = xi - xj;
		dp[idx] += xij*wSmooth(square(len_xij), square(sph._delta))/len_xij;
	}
}
PYTHON()
void sphAddPositionCorrection(
	ParticleDataImpl<Vec3> &dp, const BasicParticleSystem &x, const SphWorld &sph,
	const int itype=SphWorld::FlagFluid,
	const int jtype=SphWorld::FlagFluid|SphWorld::FlagObstacle) {
	knSphAddPositionCorrection(dp, x, sph, itype, jtype);
}

PYTHON()
int sphCountParticles(const ParticleDataImpl<int> &t, const int type) {
	int cnt=0;
	for(int i=0; i<t.size(); ++i) if(t[i]&type) ++cnt;
	return cnt;
}

} // namespace
