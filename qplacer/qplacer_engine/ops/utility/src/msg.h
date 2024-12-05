#ifndef QPLACER_UTILITY_MSG_H
#define QPLACER_UTILITY_MSG_H

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include "utility/src/namespace.h"

QPLACER_BEGIN_NAMESPACE

/// message type for print functions
enum MessageType {
  kNONE = 0,
  kINFO = 1,
  kWARN = 2,
  kERROR = 3,
  kDEBUG = 4,
  kASSERT = 5
};

/// print to screen (stdout)
int qplacerPrint(MessageType m, const char* format, ...);
/// print to stream
int qplacerPrintStream(MessageType m, FILE* stream, const char* format, ...);
/// core function to print formatted data from variable argument list
int qplacerVPrintStream(MessageType m, FILE* stream, const char* format,
                           va_list args);
/// format to a buffer
int qplacerSPrint(MessageType m, char* buf, const char* format, ...);
/// core function to format a buffer
int qplacerVSPrint(MessageType m, char* buf, const char* format,
                      va_list args);
/// format prefix
int qplacerSPrintPrefix(MessageType m, char* buf);

/// assertion
void qplacerPrintAssertMsg(const char* expr, const char* fileName,
                              unsigned lineNum, const char* funcName,
                              const char* format, ...);
void qplacerPrintAssertMsg(const char* expr, const char* fileName,
                              unsigned lineNum, const char* funcName);

#define qplacerAssertMsg(condition, args...)                       \
  do {                                                                \
    if (!(condition)) {                                               \
      ::QPlacer::qplacerPrintAssertMsg(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__, args); \
      abort();                                                        \
    }                                                                 \
  } while (false)

#define qplacerAssert(condition)                             \
  do {                                                          \
    if (!(condition)) {                                         \
      ::QPlacer::qplacerPrintAssertMsg(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
      abort();                                                  \
    }                                                           \
  } while (false)

/// static assertion
template <bool>
struct qplacerStaticAssert;
template <>
struct qplacerStaticAssert<true> {
  qplacerStaticAssert(const char* = NULL) {}
};

QPLACER_END_NAMESPACE

#endif
