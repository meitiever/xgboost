/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#ifndef COMMON_TIME_H
#define COMMON_TIME_H

/*********************************************************************
 ** Pragmas
 *********************************************************************/

#ifdef _MSC_VER
// Rostime has some magic interface that doesn't directly include
// its implementation, this just disables those warnings.
#pragma warning(disable : 4244)
#pragma warning(disable : 4661)
#endif

/*********************************************************************
 ** Headers
 *********************************************************************/

#include <cmath>
#include <iostream>
#include <stdexcept>
#include "duration.h"

/*********************************************************************
 ** Cross Platform Headers
 *********************************************************************/

#if defined(_WIN32)
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

namespace ros {

/*********************************************************************
   ** Exceptions
   *********************************************************************/

/**
   * @brief Thrown if the ros subsystem hasn't been initialised before use.
   */
class TimeNotInitializedException : public std::runtime_error {
 public:
  TimeNotInitializedException()
      : std::runtime_error(
            "Cannot use ros::Time::now() before the first NodeHandle has been created or ros::start() has been called.  "
            "If this is a standalone app or test that just uses ros::Time and does not communicate over ROS, you may also call ros::Time::init()") {}
};

/**
   * @brief Thrown if windows high perf. timestamping is unavailable.
   *
   * @sa getWallTime
   */
class NoHighPerformanceTimersException : public std::runtime_error {
 public:
  NoHighPerformanceTimersException()
      : std::runtime_error("This windows platform does not support the high-performance timing api.") {}
};

/*********************************************************************
   ** Functions
   *********************************************************************/

inline void normalizeSecNSec(uint64_t& sec, uint64_t& nsec) {
  uint64_t nsec_part = nsec % 1000000000UL;
  uint64_t sec_part  = nsec / 1000000000UL;

  if (sec + sec_part > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("Time is out of dual 32-bit range");

  sec += sec_part;
  nsec = nsec_part;
}

inline void normalizeSecNSec(uint32_t& sec, uint32_t& nsec) {
  uint64_t sec64  = sec;
  uint64_t nsec64 = nsec;

  normalizeSecNSec(sec64, nsec64);

  sec  = (uint32_t)sec64;
  nsec = (uint32_t)nsec64;
}

inline void normalizeSecNSecUnsigned(int64_t& sec, int64_t& nsec) {
  int64_t nsec_part = nsec % 1000000000L;
  int64_t sec_part  = sec + nsec / 1000000000L;
  if (nsec_part < 0) {
    nsec_part += 1000000000L;
    --sec_part;
  }

  if (sec_part < 0 || sec_part > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("Time is out of dual 32-bit range");

  sec  = sec_part;
  nsec = nsec_part;
}

/*********************************************************************
   ** Time Classes
   *********************************************************************/

/**
   * \brief Base class for Time implementations.  Provides storage, common functions and operator overloads.
   * This should not need to be used directly.
   */
template <class T, class D>
class TimeBase {
 public:
  uint32_t sec, nsec;

  TimeBase() : sec(0), nsec(0) {}
  TimeBase(uint32_t _sec, uint32_t _nsec) : sec(_sec), nsec(_nsec) {
    normalizeSecNSec(sec, nsec);
  }
  explicit TimeBase(double t) { fromSec(t); }
  D operator-(const T& rhs) const {
    D d;
    return d.fromNSec(toNSec() - rhs.toNSec());
  }
  T operator+(const D& rhs) const {
    int64_t sec_sum  = static_cast<uint64_t>(sec) + static_cast<uint64_t>(rhs.sec);
    int64_t nsec_sum = static_cast<uint64_t>(nsec) + static_cast<uint64_t>(rhs.nsec);

    // Throws an exception if we go out of 32-bit range
    normalizeSecNSecUnsigned(sec_sum, nsec_sum);

    // now, it's safe to downcast back to uint32 bits
    return T(static_cast<uint32_t>(sec_sum), static_cast<uint32_t>(nsec_sum));
  }
  T operator-(const D& rhs) const {
    return *static_cast<const T*>(this) + (-rhs);
  }
  T& operator+=(const D& rhs) {
    *this = *this + rhs;
    return *static_cast<T*>(this);
  }
  T& operator-=(const D& rhs) {
    *this += (-rhs);
    return *static_cast<T*>(this);
  }
  bool operator==(const T& rhs) const {
    return sec == rhs.sec && nsec == rhs.nsec;
  }
  inline bool operator!=(const T& rhs) const { return !(*static_cast<const T*>(this) == rhs); }
  bool operator>(const T& rhs) const {
    if (sec > rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec > rhs.nsec)
      return true;
    return false;
  }
  bool operator<(const T& rhs) const {
    if (sec < rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec < rhs.nsec)
      return true;
    return false;
  }
  bool operator>=(const T& rhs) const {
    if (sec > rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec >= rhs.nsec)
      return true;
    return false;
  }
  bool operator<=(const T& rhs) const {
    if (sec < rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec <= rhs.nsec)
      return true;
    return false;
  }

  double toSec() const { return static_cast<double>(sec) + 1e-9 * static_cast<double>(nsec); };
  T& fromSec(double t) {
    int64_t sec64 = static_cast<int64_t>(floor(t));
    if (sec64 < 0 || sec64 > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("Time is out of dual 32-bit range");
    sec  = static_cast<uint32_t>(sec64);
    nsec = static_cast<uint32_t>(std::round((t - sec) * 1e9));
    // avoid rounding errors
    sec += (nsec / 1000000000ul);
    nsec %= 1000000000ul;
    return *static_cast<T*>(this);
  }

  uint64_t toNSec() const { return static_cast<uint64_t>(sec) * 1000000000ull + static_cast<uint64_t>(nsec); }
  T& fromNSec(uint64_t t) {
    uint64_t sec64  = 0;
    uint64_t nsec64 = t;

    normalizeSecNSec(sec64, nsec64);

    sec  = static_cast<uint32_t>(sec64);
    nsec = static_cast<uint32_t>(nsec64);

    return *static_cast<T*>(this);
  }

  inline bool isZero() const { return sec == 0 && nsec == 0; }
  inline bool is_zero() const { return isZero(); }
};

/**
   * \brief Time representation.  May either represent wall clock time or ROS clock time.
   *
   * ros::TimeBase provides most of its functionality.
   */
class Time : public TimeBase<Time, Duration> {
 public:
  Time() : TimeBase<Time, Duration>() {}

  Time(uint32_t _sec, uint32_t _nsec) : TimeBase<Time, Duration>(_sec, _nsec) {}

  explicit Time(double t) { fromSec(t); }
};

const Time TIME_MAX(std::numeric_limits<uint32_t>::max(), 999999999);
const Time TIME_MIN(0, 1);

std::ostream& operator<<(std::ostream& os, const Time& rhs);
}

#endif  // COMMON_TIME_H
