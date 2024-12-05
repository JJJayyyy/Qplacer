#ifndef QPLACER_UTILITY_TIMER_H
#define QPLACER_UTILITY_TIMER_H

#include <chrono>
#include "utility/src/namespace.h"

QPLACER_BEGIN_NAMESPACE

struct CPUTimer {
  typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

  static inline hr_clock_rep getGlobaltime(void) {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }

  // Returns the period in miliseconds
  static inline double getTimerPeriod(void) {
    return 1000.0 * std::chrono::high_resolution_clock::period::num /
           std::chrono::high_resolution_clock::period::den;
  }
};

QPLACER_END_NAMESPACE

#endif
