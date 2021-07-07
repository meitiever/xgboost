/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
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

#ifndef COMMON_DURATION_H
#define COMMON_DURATION_H

/*********************************************************************
 ** Pragmas
 *********************************************************************/

#ifdef _MSC_VER
// Rostime has some magic interface that doesn't directly include
// its implementation, this just disbales those warnings.
#pragma warning(disable : 4244)
#pragma warning(disable : 4661)
#endif

#include <math.h>
#include <stdint.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ros {
inline void normalizeSecNSecSigned(int64_t &sec, int64_t &nsec) {
  int64_t nsec_part = nsec % 1000000000L;
  int64_t sec_part  = sec + nsec / 1000000000L;
  if (nsec_part < 0) {
    nsec_part += 1000000000L;
    --sec_part;
  }

  if (sec_part < std::numeric_limits<int32_t>::min() || sec_part > std::numeric_limits<int32_t>::max())
    throw std::runtime_error("Duration is out of dual 32-bit range");

  sec  = sec_part;
  nsec = nsec_part;
}

inline void normalizeSecNSecSigned(int32_t &sec, int32_t &nsec) {
  int64_t sec64  = sec;
  int64_t nsec64 = nsec;

  normalizeSecNSecSigned(sec64, nsec64);

  sec  = (int32_t)sec64;
  nsec = (int32_t)nsec64;
}

/**
 * \brief Base class for Duration implementations.  Provides storage, common functions and operator overloads.
 * This should not need to be used directly.
 */
template <class T>
class DurationBase {
 public:
  int32_t sec, nsec;
  DurationBase() : sec(0), nsec(0) {}
  DurationBase(int32_t _sec, int32_t _nsec) : sec(_sec), nsec(_nsec) {
    normalizeSecNSecSigned(sec, nsec);
  }

  explicit DurationBase(double t) { fromSec(t); };

  ~DurationBase() {}

  T operator+(const T &rhs) const {
    T t;
    return t.fromNSec(toNSec() + rhs.toNSec());
  }

  T operator-(const T &rhs) const {
    T t;
    return t.fromNSec(toNSec() - rhs.toNSec());
  }

  T operator-() const {
    T t;
    return t.fromNSec(-toNSec());
  }

  T operator*(double scale) const {
    return T(toSec() * scale);
  }

  T &operator+=(const T &rhs) {
    *this = *this + rhs;
    return *static_cast<T *>(this);
  }

  T &operator-=(const T &rhs) {
    *this += (-rhs);
    return *static_cast<T *>(this);
  }

  T &operator*=(double scale) {
    fromSec(toSec() * scale);
    return *static_cast<T *>(this);
  }

  bool operator==(const T &rhs) const {
    return sec == rhs.sec && nsec == rhs.nsec;
  }

  inline bool operator!=(const T &rhs) const { return !(*static_cast<const T *>(this) == rhs); }

  bool operator>(const T &rhs) const {
    if (sec > rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec > rhs.nsec)
      return true;
    return false;
  }

  bool operator<(const T &rhs) const {
    if (sec < rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec < rhs.nsec)
      return true;
    return false;
  }

  bool operator>=(const T &rhs) const {
    if (sec > rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec >= rhs.nsec)
      return true;
    return false;
  }

  bool operator<=(const T &rhs) const {
    if (sec < rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec <= rhs.nsec)
      return true;
    return false;
  }

  double toSec() const { return static_cast<double>(sec) + 1e-9 * static_cast<double>(nsec); };

  int64_t toNSec() const { return static_cast<int64_t>(sec) * 1000000000ll + static_cast<int64_t>(nsec); };

  T &fromSec(double d) {
    int64_t sec64 = static_cast<int64_t>(floor(d));
    if (sec64 < std::numeric_limits<int32_t>::min() || sec64 > std::numeric_limits<int32_t>::max())
      throw std::runtime_error("Duration is out of dual 32-bit range");
    sec              = static_cast<int32_t>(sec64);
    nsec             = static_cast<int32_t>(std::round((d - sec) * 1e9));
    int32_t rollover = nsec / 1000000000ul;
    sec += rollover;
    nsec %= 1000000000ul;
    return *static_cast<T *>(this);
  }

  T &fromNSec(int64_t t) {
    int64_t sec64 = t / 1000000000LL;
    if (sec64 < std::numeric_limits<int32_t>::min() || sec64 > std::numeric_limits<int32_t>::max())
      throw std::runtime_error("Duration is out of dual 32-bit range");
    sec  = static_cast<int32_t>(sec64);
    nsec = static_cast<int32_t>(t % 1000000000LL);

    normalizeSecNSecSigned(sec, nsec);

    return *static_cast<T *>(this);
  }

  bool isZero() const {
    return sec == 0 && nsec == 0;
  }
};

/**
 * \brief Duration representation for use with the Time class.
 *
 * ros::DurationBase provides most of its functionality.
 */
class Duration : public DurationBase<Duration> {
 public:
  Duration() : DurationBase<Duration>() {}

  Duration(int32_t _sec, int32_t _nsec) : DurationBase<Duration>(_sec, _nsec) {}

  explicit Duration(double t) { fromSec(t); }
};

//extern const Duration DURATION_MAX;
//extern const Duration DURATION_MIN;

std::ostream &operator<<(std::ostream &os, const Duration &rhs);
}

#endif  // COMMON_DURATION_H
