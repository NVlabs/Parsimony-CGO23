#!/bin/bash

set -e

if [ -z "${PARSIM_ROOT}" ]
then
	echo "Please set PARSIM_ROOT to the root of the Parsimony repository."
	exit 1
fi

if [ -z "${SETUP_ENVIRONMENT_WAS_RUN}" ]
then
	echo "Please source ${PARSIM_ROOT}/scripts/setup_environment.sh"
	exit 1
fi

if [ `clang++ --version | head -n 1 | awk -F'(' '{print $1}' | awk '{print $3}' | awk -F'.' '{print $1}'` != "15" ]; 
then 
	echo "Error: LLVM-15 not installed.";
	exit 1
fi

if [ `ispc --version | awk -F'(' '{print $4}' | awk -F'), ' '{print $2}'` != "1.18.0" ];
then
	echo "Error: ispc 1.18.0 not installed";
	exit 1
fi

COMPILER_DIR=${PARSIM_ROOT}/compiler
SIMD_DIR=${PARSIM_ROOT}/apps/synet-simd
ISPC_DIR=${PARSIM_ROOT}/apps/ispc-benchs
RESULTS_DIR=${PARSIM_ROOT}/results

## PARSIMONY
echo " "
echo "----------------- Installing Parsimony ------------------"
echo " "
#export PARSIM_INSTALL_PATH=${COMPILER_DIR}/install
cd ${COMPILER_DIR}
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_INSTALL_DIR=${LLVM_ROOT} -DSLEEF_INSTALL_DIR=${SLEEF_ROOT} -DCMAKE_INSTALL_PREFIX=${PARSIM_INSTALL_PATH} -DZ3_INSTALL_DIR=${Z3_ROOT} ..
cmake --build . -j 16 --target install

#export PATH=${PARSIM_INSTALL_PATH}/bin:${PATH}
cd ${PARSIM_ROOT}


## Simd Library
echo " "
echo "----------------- Installing Simd Library Benchmarks ------------------"
echo " "
. ${SIMD_DIR}/scripts/build_simdlib.sh


## ispc-benchs
echo " "
echo "----------------- Installing ispc Benchmarks ------------------"
echo " "
. ${ISPC_DIR}/scripts/build_ispc.sh
echo " "
echo "----------------- DONE ------------------"
