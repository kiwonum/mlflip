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

#include "pneighbors.h"

namespace Manta {

KERNEL(pts)
void knUpdateNeighbor(
	ParticleNeighbors::NeighborData &neighbor,
	const BasicParticleSystem &pts,	const Grid<int> &index, const ParticleIndexSystem &indexSys, const Real radius,
	const ParticleDataImpl<int> *pT, const int exclude) {
	if(!pts.isActive(idx) || (pT && (*pT)[idx]&exclude)) return;
	const Vec3 &xi = pts[idx].pos;
	const Vec3i xidx = toVec3i(xi);
	if(!index.isInBounds(xidx)) return;

	// search cells around each particle, idx
	neighbor[idx].clear();
	const Real radiusSqr = square(radius);
	const int r  = static_cast<int>(radius) + 1;
	const int rZ = pts.getParent()->is3D() ? r : 0;
	for(int k=xidx.z-rZ; k<=xidx.z+rZ; ++k) {
		for(int j=xidx.y-r; j<=xidx.y+r; ++j) {
			for(int i=xidx.x-r; i<=xidx.x+r; ++i) {
				if(!index.isInBounds(Vec3i(i, j, k))) continue;

				// loop for particles in each cell
				const int isysIdxS = index.index(i, j, k);
				const int pStart = index(isysIdxS);
				const int pEnd = index.isInBounds(isysIdxS+1) ? index(isysIdxS+1) : indexSys.size();
				for(int p=pStart; p<pEnd; ++p) {
					const int psrc = indexSys[p].sourceIndex;
					const Vec3 &xj = pts[psrc].pos;
					const Real lensqr_xij = normSquare(xi - xj);
					if(lensqr_xij>radiusSqr) continue;
					neighbor[idx].push_back(std::make_pair(psrc, std::sqrt(lensqr_xij)));
				}
			}
		}
	}
}
void ParticleNeighbors::update(
	const BasicParticleSystem &pts, const Grid<int> &index, const ParticleIndexSystem &indexSys, const Real radius,
	const ParticleDataImpl<int> *pT, const int exclude) {
	_neighbor.resize(pts.size());
	knUpdateNeighbor(_neighbor, pts, index, indexSys, radius, pT, exclude);
}

KERNEL(pts)
void knUpdateDistance(ParticleNeighbors::NeighborData &neighbor, const BasicParticleSystem &pts) {
	for(unsigned int jj=0; jj<neighbor[idx].size(); ++jj) {
		const int j = neighbor[idx][jj].first;
		const Real len_xij = norm(pts[idx].pos - pts[j].pos);
		neighbor[idx][jj].second = len_xij;
	}
}
void ParticleNeighbors::updateDistance(const BasicParticleSystem &pts) {
	knUpdateDistance(_neighbor, pts);
}

} // namespace
