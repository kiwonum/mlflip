/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2015-2016 Kiwon Um, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL)
 * http://www.gnu.org/licenses
 *
 * SPH Kernels
 *
 ******************************************************************************/

#ifndef SPHKERNEL_H
#define SPHKERNEL_H

#include "manta.h"

namespace Manta {

PYTHON()
class CubicSpline : public PbClass {
public:
	PYTHON() CubicSpline(FluidSolver *parent, const Real h=1) : PbClass(parent) {
		_dim = (parent->is2D()) ? 2 : 3;
		setRadius(h);
	}
	void setRadius(const Real h) {
		const Real h2 = square(h), h3 = h2*h, h4 = h3*h, h5 = h4*h;
		_h = h;
		_sr = 2e0*h;
		_c[0]  = 2e0/(3e0*h);
		_c[1]  = 10e0/(7e0*M_PI*h2);
		_c[2]  = 1e0/(M_PI*h3);
		_gc[0] = 3e0/(2e0*h3);
		_gc[1] = 45e0/(14e0*M_PI*h4);
		_gc[2] = 9e0/(4e0*M_PI*h5);
	}
	PYTHON() Real radius() const { return _h; }
	PYTHON() Real supportRadius() const { return _sr; }

	Real f(const Real l) const {
		const Real q = l/_h;
		if(q<1e0) return _c[_dim-1]*(1e0 - 1.5*square(q) + 0.75*cubed(q));
		else if(q<2e0) return _c[_dim-1]*(0.25*cubed(2e0-q));
		return 0;
	}
	Real derivative_f(const Real l) const {
		const Real q = l/_h;
		if(q<=1e0) return _gc[_dim-1]*(q-4e0/3e0)*l;
		else if(q<2e0) return -_gc[_dim-1]*square(2e0-q)*_h/3e0;
		return 0;
	}

	Real w(const Vec3 &rij) const { return f(norm(rij)); }
	Vec3 grad_w(const Vec3 &rij) const { return grad_w(rij, norm(rij)); }
	Vec3 grad_w(const Vec3 &rij, const Real len) const { return derivative_f(len)*rij/len; }

private:
	unsigned int _dim;
	Real _h, _sr, _c[3], _gc[3];
};

PYTHON()
class BndKernel : public PbClass {
public:
	PYTHON() BndKernel(FluidSolver *parent, const Real c, const Real h=1) : PbClass(parent), _c(c) {
		setRadius(h);
	}
	void setRadius(const Real h) {
		_h = h;
		_sr = 2e0*h;
	}
	Real radius() const { return _h; }
	Real supportRadius() const { return _sr; }

	Real f(const Real l) const {
		const Real q = l/_h;
		const Real s = 0.02*square(_c)/l;
		if(q<=2e0/3e0) return s*2e0/3e0;
		else if(q<1e0) return s*(2e0*q - square(q)*2e0/3e0);
		else if(q<2e0) return s*0.5*square(2e0 - q);
		return 0e0;
	}

private:
	Real _h, _sr, _c;
};

inline Real wSmooth(const Real &r2, const Real h2) { return (r2>h2) ? 0.0 : 1.0 - r2/h2; }

} // namespaces

#endif	/* SPHKERNEL_H */
