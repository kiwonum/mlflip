/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Plugin timing
 *
 ******************************************************************************/

#ifndef _TIMING_H
#define _TIMING_H

#include "manta.h"
#include <map>
namespace Manta {


class TimingData {
private:
	TimingData();
public:
	static TimingData& instance() { static TimingData a; return a; }

	void print();
	void saveMean(const std::string& filename);
	void start(FluidSolver* parent, const std::string& name);
	void stop(FluidSolver* parent, const std::string& name);
	void step();

protected:
	struct TimingSet {
		TimingSet() : num(0), updated(false) { cur.clear(); total.clear(); }
		MuTime cur, total;
		int num;
		bool updated;
		std::string solver;
	};
	bool updated;

	int num;
	MuTime mPluginTimer;
	std::string mLastPlugin;
	std::map< std::string, std::vector<TimingSet> > mData;
};

// Python interface
PYTHON() class Timings : public PbClass {
public:
	PYTHON() Timings() : PbClass(NULL) {}

	PYTHON() void display() { TimingData::instance().print(); }
	PYTHON() void step() { TimingData::instance().step(); }
	PYTHON() void saveMean(const std::string& file) { TimingData::instance().saveMean(file); }
};

PYTHON() class Timer : public PbClass {
public:
	PYTHON() Timer() : PbClass(NULL) {}

	PYTHON() void push(const std::string &name) {
		mName.push_back(name);
		mTime.push_back(MuTime());
	}
	PYTHON() void pop(const std::string &name) {
		mPopTiming = MuTime() - mTime.back();
		mPopName = mName.back();
		assertMsg(name==mPopName, "incorrect use of timer; you must pop " + mPopName);
		mName.pop_back();
		mTime.pop_back();
	}
	PYTHON() const std::string& popName() const { return mPopName; }
	PYTHON() int popTiming() const { return mPopTiming.time; }
	PYTHON() std::string popTimingStr() const { return mPopTiming.toString(); }

private:
	std::vector<std::string> mName;
	std::vector<MuTime> mTime;

	MuTime mPopTiming;
	std::string mPopName;
};

}

#endif
