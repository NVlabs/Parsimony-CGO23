#!/bin/bash

# export LLVM_ROOT=${PARSIM_ROOT}/llvm-project/install
# export PATH=${LLVM_ROOT}/bin:${PATH} # Adding LLVM binaries to $PATH
# export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:${LD_LIBRARY_PATH} # Adding LLVM libraries to $LD_LIBRARY_PATH
export PATH=${PARSIM_ROOT}/ispc-v1.18.0-linux/bin:${PATH} # Adding ispc binaries to $PATH
export Z3_ROOT=${PARSIM_ROOT}/z3/install
export SLEEF_ROOT=${PARSIM_ROOT}/sleef/install
export PARSIM_INSTALL_PATH=${PARSIM_ROOT}/compiler/install
export PATH=${PARSIM_INSTALL_PATH}/bin:${PATH} # Adding Parsimony binary to $PATH
export SETUP_ENVIRONMENT_WAS_RUN=1