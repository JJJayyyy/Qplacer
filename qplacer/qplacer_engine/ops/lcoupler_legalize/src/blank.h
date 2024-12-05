#ifndef QPLACER_BLANK_H
#define QPLACER_BLANK_H

#include "utility/src/utils.h"

QPLACER_BEGIN_NAMESPACE

template <typename T>
struct Interval {
	T xl;
	T xh;

	Interval(T l, T h) : xl(l), xh(h) {}

	void intersect(T rhs_xl, T rhs_xh) {
		xl = std::max(xl, rhs_xl);
		xh = std::min(xh, rhs_xh);
	}

	std::string toString() const {
		return "Interval(" + std::to_string(xl) + ", " + std::to_string(xh) + ")";
	}
};

template <typename T>
struct Blank {
	T xl;
	T yl;
	T xh;
	T yh;

	void intersect(const Blank& rhs) {
		xl = std::max(xl, rhs.xl);
		xh = std::min(xh, rhs.xh);
		yl = std::max(yl, rhs.yl);
		yh = std::min(yh, rhs.yh);
	}

	std::string toString() const {
		return "Blank(" + std::to_string(xl) + ", " + std::to_string(yl) + ", " 
						+ std::to_string(xh) + ", " + std::to_string(yh) + ")";
	}
};

QPLACER_END_NAMESPACE

#endif
