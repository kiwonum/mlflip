// ----------------------------------------------------------------------------
//
// MantaFlow fluid solver framework
// Copyright 2015-2016 Kiwon Um, Nils Thuerey
//
// This program is free software, distributed under the terms of the
// GNU General Public License (GPL)
// http://www.gnu.org/licenses
//
// SPH simulation world
//
// ----------------------------------------------------------------------------

#ifndef SPHSOLVER_H
#define SPHSOLVER_H

#include "manta.h"

#include "particle.h"
#include "pneighbors.h"
#include "vectorbase.h"

namespace Manta {

PYTHON()
class SphWorld : public PbClass {
public:
	enum { FlagFluid=FlagGrid::TypeFluid, FlagObstacle=FlagGrid::TypeObstacle };

	static Real equationOfState(const Real d, const Real d0, const Real p0, const Real gamma, const Real chi=0.0) {
		return p0*(std::pow(d/d0, gamma) - 1.0) + chi;
	}

	PYTHON() SphWorld(
		FluidSolver *parent,
		const Real alpha=0.08, const Real delta=0.5, const Real density=1e3,
		const Real eta=0.01, const Vec3 g=Vec3(0,-9.8,0), const Real gamma=7.0) : PbClass(parent), _alpha(alpha), _eta(eta), _g(g) {
		updateDelta(delta);
		updateDensity(density);
		updateGamma(gamma);
		updateSoundSpeed(static_cast<Real>(parent->getGridSize().y)*std::fabs(_g.y)/_eta);
	}

	PYTHON() void bindParticleSystem(const BasicParticleSystem &p_system, const ParticleDataImpl<int> &p_type, const ParticleNeighbors &p_neighbor) {
		_pSystem = &p_system;
		_type = &p_type;
		_neighbor = &p_neighbor;
	}

	PYTHON() void updateDelta(const Real d) { _delta = d; update_m0(); }
	PYTHON() void updateDensity(const Real d) { _d0 = d; update_m0(); update_p0(); }
	PYTHON() void updateSoundSpeed(const Real c) { _c = c; update_p0(); }
	PYTHON() void updateGamma(const Real gamma) { _gamma = gamma; update_p0(); }

	const BasicParticleSystem &particleSystem() const { return *_pSystem; }
	const ParticleDataImpl<int> &particleType() const { return *_type; }
	const ParticleNeighbors &neighborData() const { return *_neighbor; }

	PYTHON() Real limitDtByVmax(const Real dt, const Real h, const Real vmax, const Real a=0.4) const {
		return std::min(dt, a*h/(vmax+VECTOR_EPSILON));
	}
	PYTHON() Real limitDtByFmax(const Real dt, const Real h, const Real fmax, const Real a=0.25) const {
		return std::min(dt, a*std::sqrt(h/(fmax+VECTOR_EPSILON)));
	}
	PYTHON() Real limitDtByGravity(const Real dt, const Real h, const Real a=0.25) const {
		const Real norm_g = norm(_g);
		return (norm_g<=VECTOR_EPSILON) ? dt : std::min(dt, a*std::sqrt(h/norm_g));
	}
	PYTHON() Real limitDtByViscous(const Real dt, const Real h, const Real a=0.4) const { // WCSPH [Monaghan 1992, Becker and Teschner 2007]
		return std::min(dt, a*h/(_c*(static_cast<Real>(1.0)+static_cast<Real>(0.6)*_alpha)));
	}

	PYTHON() void showParameters() const {
		std::cout << "SPH parameters:" << std::endl;
		std::cout << "\t" << "alpha = " << _alpha << std::endl;
		std::cout << "\t" << "c = " << _c << std::endl;
		std::cout << "\t" << "d0 = " << _d0 << std::endl;
		std::cout << "\t" << "delta = " << _delta << std::endl;
		std::cout << "\t" << "eta = " << _eta << std::endl;
		std::cout << "\t" << "g = " << _g << std::endl;
		std::cout << "\t" << "gamma = " << _gamma << std::endl;
		std::cout << "\t" << "m0 = " << _m0 << std::endl;
		std::cout << "\t" << "p0 = " << _p0 << std::endl;
	}

	PYTHON(name=alpha)    Real _alpha;
	PYTHON(name=c)	      Real _c;
	PYTHON(name=density)  Real _d0;
	PYTHON(name=delta)    Real _delta; // particle spacing
	PYTHON(name=eta)      Real _eta;
	PYTHON(name=gravity)  Vec3 _g;
	PYTHON(name=gamma)    Real _gamma;
	PYTHON(name=mass)     Real _m0;
	PYTHON(name=pressure) Real _p0;

private:
	void update_p0() { _p0 = _d0*_c*_c/_gamma; }
	void update_m0() { _m0 = _d0*std::pow(_delta, this->getParent()->is2D() ? 2.0 : 3.0); }

	// mandatory data
	const BasicParticleSystem *_pSystem;
	const ParticleDataImpl<int> *_type;
	const ParticleNeighbors *_neighbor;
};

// Functors
struct Pij {			// [Gingold and Monaghan, 1982, JCP]
	explicit Pij(const ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d, const Real m, const ParticleDataImpl<int> &t) : _p(p), _d(d), _t(t), _mSqr(square(m)) {}
	Real operator()(const int i, int j) const { j=((_t[j]&SphWorld::FlagObstacle)?i:j); return _mSqr*(_p[i]/square(_d[i]) + _p[j]/square(_d[j])); }
	const ParticleDataImpl<Real> &_p, &_d;
	const ParticleDataImpl<int> &_t;
	const Real _mSqr;
};
struct PijMij {			// [Gingold and Monaghan, 1982, JCP] + [Akinci et al., 2012, SIG/TOG]
	explicit PijMij(const ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> &m, const ParticleDataImpl<int> &t) : _p(p), _d(d), _m(m), _t(t) {}
	Real operator()(const int i, int j) const { j=((_t[j]&SphWorld::FlagObstacle)?i:j); return _m[i]*_m[j]*(_p[i]/square(_d[i]) + _p[j]/square(_d[j])); }
	const ParticleDataImpl<Real> &_p, &_d, &_m;
	const ParticleDataImpl<int> &_t;
};
struct PijDfsph {		// [Bender and Koschier, 2015, SCA]
	explicit PijDfsph(const ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d, const Real m, const ParticleDataImpl<int> &t) : _p(p), _d(d), _t(t), _mSqr(square(m)) {}
	Real operator()(const int i, int j) const { j=((_t[j]&SphWorld::FlagObstacle)?i:j); return _mSqr*(_p[i]/_d[i] + _p[j]/_d[j]); }
	const ParticleDataImpl<Real> &_p, &_d;
	const ParticleDataImpl<int> &_t;
	const Real _mSqr;
};
struct PijMijDfsph {		// [Bender and Koschier, 2015, SCA] + [Akinci et al., 2012, SIG/TOG]
	explicit PijMijDfsph(const ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> &m, const ParticleDataImpl<int> &t) : _p(p), _d(d), _m(m), _t(t) {}
	Real operator()(const int i, int j) const { j=((_t[j]&SphWorld::FlagObstacle)?i:j); return _m[i]*_m[j]*(_p[i]/_d[i] + _p[j]/_d[j]); }
	const ParticleDataImpl<Real> &_p, &_d, &_m;
	const ParticleDataImpl<int> &_t;
};
struct PijWeighted {		// [Hu and Adams, 2007, JCP]
	explicit PijWeighted(const ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &d, const ParticleDataImpl<Real> &Vsqr) : _p(p), _d(d), _Vsqr(Vsqr) {}
	Real operator()(const int i, const int j) const { return (_Vsqr[i]+_Vsqr[j])*(_p[i]*_d[j] + _p[j]*_d[i])/(_d[i]+_d[j]); }
	const ParticleDataImpl<Real> &_p, &_d, &_Vsqr;
};
struct PijBg {
	explicit PijBg(const ParticleDataImpl<Real> &p, const ParticleDataImpl<Real> &Vsqr) : _p(p), _Vsqr(Vsqr) {}
	Real operator()(const int i, const int j) const { return 10e0*std::fabs(_p[i])*(_Vsqr[i]+_Vsqr[j]); }
	const ParticleDataImpl<Real> &_p, &_Vsqr;
};

} // namespace

#endif	/* SPHSOLVER_H */
