#!/bin/bash
COMPILER_DIR=${PARSIM_ROOT}/compiler
SIMD_DIR=${PARSIM_ROOT}/apps/synet-simd
ISPC_DIR=${PARSIM_ROOT}/apps/ispc-benchs
RESULTS_DIR=${PARSIM_ROOT}/results

rm -r ${COMPILER_DIR}/install
rm -r ${COMPILER_DIR}/build
${SIMD_DIR}/scripts/build_simdlib.sh clean
${ISPC_DIR}/scripts/build_ispc.sh clean
rm ${RESULTS_DIR}/*.csv
unset PARSIMONY_AND_APPS_INSTALLED
unset PARSIM_INSTALL_PATH
echo " "
echo "----------------- DONE ------------------"
