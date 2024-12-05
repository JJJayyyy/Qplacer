
#include "utility/src/msg.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

QPLACER_BEGIN_NAMESPACE

int qplacerPrint(MessageType m, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int ret = qplacerVPrintStream(m, stdout, format, args);
  va_end(args);

  return ret;
}

int qplacerPrintStream(MessageType m, FILE* stream, const char* format,
                          ...) {
  va_list args;
  va_start(args, format);
  int ret = qplacerVPrintStream(m, stream, format, args);
  va_end(args);

  return ret;
}

int qplacerVPrintStream(MessageType m, FILE* stream, const char* format,
                           va_list args) {
  // print prefix
  char prefix[16];
  qplacerSPrintPrefix(m, prefix);
  fprintf(stream, "%s", prefix);

  // print message
  int ret = vfprintf(stream, format, args);

  return ret;
}

int qplacerSPrint(MessageType m, char* buf, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int ret = qplacerVSPrint(m, buf, format, args);
  va_end(args);

  return ret;
}

int qplacerVSPrint(MessageType m, char* buf, const char* format,
                      va_list args) {
  // print prefix
  char prefix[16];
  qplacerSPrintPrefix(m, prefix);
  sprintf(buf, "%s", prefix);

  // print message
  int ret = vsprintf(buf + strlen(prefix), format, args);

  return ret;
}

int qplacerSPrintPrefix(MessageType m, char* prefix) {
  switch (m) {
    case kNONE:
      return sprintf(prefix, "%c", '\0');
    case kINFO:
      return sprintf(prefix, "[INFO   ] ");
    case kWARN:
      return sprintf(prefix, "[WARNING] ");
    case kERROR:
      return sprintf(prefix, "[ERROR  ] ");
    case kDEBUG:
      return sprintf(prefix, "[DEBUG  ] ");
    case kASSERT:
      return sprintf(prefix, "[ASSERT ] ");
    default:
      qplacerAssertMsg(0, "unknown message type");
  }
  return 0;
}

void qplacerPrintAssertMsg(const char* expr, const char* fileName,
                              unsigned lineNum, const char* funcName,
                              const char* format, ...) {
  // construct message
  char buf[1024];
  va_list args;
  va_start(args, format);
  vsprintf(buf, format, args);
  va_end(args);

  // print message
  qplacerPrintStream(kASSERT, stderr,
                        "%s:%u: %s: Assertion `%s' failed: %s\n", fileName,
                        lineNum, funcName, expr, buf);
}

void qplacerPrintAssertMsg(const char* expr, const char* fileName,
                              unsigned lineNum, const char* funcName) {
  // print message
  qplacerPrintStream(kASSERT, stderr, "%s:%u: %s: Assertion `%s' failed\n",
                        fileName, lineNum, funcName, expr);
}


QPLACER_END_NAMESPACE
