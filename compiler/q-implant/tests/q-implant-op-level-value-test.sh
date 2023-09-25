#!/bin/bash

BINDIR="$1"; shift
WORKDIR="$1"; shift
TEST_DRIVER_PATH="${BINDIR}/test_driver"
TEST_RESULT_DIR="${BINDIR}/result"

TESTED=()
PASSED=()
FAILED=()

for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${WORKDIR}/${TESTCASE}"
  TEST_RESULT_FILE="${BINDIR}/${TESTCASE}"

  PASSED_TAG="${TEST_RESULT_FILE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TEST_RESULT_FILE}.log" <(
    exec 2>&1
    set -ex

    "${TEST_DRIVER_PATH}" "${TESTCASE_FILE}.circle" "${BINDIR}/${TESTCASE}_Q8/qparam.json" "${BINDIR}/${TESTCASE}_Q8/output.circle"

    if [[ $? -eq 0 ]]; then
      touch "${PASSED_TAG}"
    fi
  )

  if [[ -f "${PASSED_TAG}" ]]; then
    PASSED+=("${TESTCASE}")
  else
    FAILED+=("${TESTCASE}")
  fi
done

if [[ ${#TESTED[@]} -ne ${#PASSED[@]} ]]; then
  echo "FAILED"
  for TEST in "${FAILED[@]}"
  do
    echo "- ${TEST}"
  done
  exit 255
fi

echo "PASSED"
exit 0
