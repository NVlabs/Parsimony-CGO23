#!/bin/bash
SIMD_DIR=${PARSIM_ROOT}/apps/synet-simd
ISPC_DIR=${PARSIM_ROOT}/apps/ispc-benchs
RESULTS_DIR=${PARSIM_ROOT}/results

if [ ! "`which psv`" == "${PARSIM_ROOT}/compiler/install/bin/psv" ];
then 
	echo "Please install Parsimony with ${PARSIM_ROOT}/scripts/install_parsim_and_apps.sh before running this script."
	exit 1
fi

## Run Simd Library
echo " "
echo "----------------- Running Simd Library Benchmarks ------------------"
echo " "
. ${SIMD_DIR}/scripts/run_simdlib.sh
. ${SIMD_DIR}/scripts/parse_simdlib_results.sh psim_results.txt > ${RESULTS_DIR}/simdlib_results.csv
echo "----------------- Results written to ${RESULTS_DIR}/simdlib_results.csv ------------------"

## Run ispc-benchs
echo " "
echo "----------------- Running ispc Benchmarks ------------------"
echo " "
. ${ISPC_DIR}/scripts/run_ispc.sh all > ${RESULTS_DIR}/ispcbench_results.csv
echo "----------------- Results written to ${RESULTS_DIR}/ispcbench_results.csv ------------------"