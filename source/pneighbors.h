// ----------------------------------------------------------------------------
//
// MantaFlow fluid solver framework
// Copyright 2015 Kiwon Um, Nils Thuerey
//
// This program is free software, distributed under the terms of the
// GNU General Public License (GPL)
// http://www.gnu.org/licenses
//
// Particle-neighbors data
//
// ----------------------------------------------------------------------------

#ifndef PNEIGHBORS_H
#define PNEIGHBORS_H

#include "manta.h"
#include "particle.h"

namespace Manta {

PYTHON() class ParticleNeighbors : public PbClass {
public:
	typedef std::vector< std::pair<int,Real> > Neighbors;
	typedef std::vector<Neighbors> NeighborData;

	PYTHON() ParticleNeighbors(FluidSolver *parent) : PbClass(parent) {}

	PYTHON() void update(const BasicParticleSystem &pts, const Grid<int> &index, const ParticleIndexSystem &indexSys, const Real radius,
			     const ParticleDataImpl<int> *pT=NULL, const int exclude=0);
	PYTHON() void updateDistance(const BasicParticleSystem &pts);

	int neighborIdx(const Neighbors::const_iterator &it) const { return it->first; }
	Real length(const Neighbors::const_iterator &it) const { return it->second; }
	Neighbors::const_iterator begin(const int idx) const { return _neighbor[idx].begin(); }
	Neighbors::const_iterator end(const int idx) const { return _neighbor[idx].end(); }
	int size(const int idx) const { return _neighbor[idx].size(); }

private:
	NeighborData _neighbor;
};

} // namespace

#endif	/* PNEIGHBORS_H */
