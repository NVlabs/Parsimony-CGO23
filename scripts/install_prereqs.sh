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

# PREREQS
# if [ -d "${PARSIM_ROOT}/llvm-project" ]
# then
# 	echo " "
# 	echo "----------------- LLVM 15.0.1 already installed ------------------"
# 	echo " "
# else
# 	echo " "
# 	echo "----------------- Installing LLVM 15.0.1 ------------------"
# 	echo " "
# 	git clone --depth=1 --single-branch --branch=llvmorg-15.0.1 https://github.com/llvm/llvm-project.git
# 	cd llvm-project
# 	mkdir build
# 	cd build
# 	#export LLVM_ROOT=${PARSIM_ROOT}/llvm-project/install
# 	cmake -DLLVM_ENABLE_PROJECTS='clang' -DLLVM_TARGETS_TO_BUILD="AArch64;X86" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT} -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;openmp" ../llvm
# 	cmake --build . -j 16 --target install
# 	#export PATH=${LLVM_ROOT}/bin:${PATH}
# 	#export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:${LD_LIBRARY_PATH}
# 	cd ${PARSIM_ROOT}
# fi

if [ -d "${PARSIM_ROOT}/ispc-v1.18.0-linux" ]
then
	echo " "
	echo "----------------- ispc v1.18.0 already installed ------------------"
	echo " "
else
	echo " "
	echo "----------------- Installing ispc v1.18.0 ------------------"
	echo " "
	wget https://github.com/ispc/ispc/releases/download/v1.18.0/ispc-v1.18.0-linux.tar.gz
	tar -xvf ispc-v1.18.0-linux.tar.gz
	rm ispc-v1.18.0-linux.tar.gz
	#export PATH=${PARSIM_ROOT}/ispc-v1.18.0-linux/bin:${PATH}
fi

if [ -d "${PARSIM_ROOT}/z3" ]
then
	echo " "
	echo "----------------- Z3Prover 4.11.2 already installed ------------------"
	echo " "
else
	echo " "
	echo "----------------- Installing Z3Prover 4.11.2 ------------------"
	echo " "
	git clone https://github.com/Z3Prover/z3.git
	cd z3
	git checkout tags/z3-4.11.2
	#export Z3_ROOT=${PARSIM_ROOT}/z3/install
	CXX=clang++ CC=clang python scripts/mk_make.py --prefix=${Z3_ROOT}
	cd build
	make -j 16
	make install
	cd ${PARSIM_ROOT}
fi

if [ -d "${PARSIM_ROOT}/sleef" ]
then
	echo " "
	echo "----------------- Sleef 3.5.1 already installed ------------------"
	echo " "
else
	echo " "
	echo "----------------- Installing Sleef 3.5.1 ------------------"
	echo " "
	git clone https://github.com/shibatch/sleef.git
	cd sleef
	git checkout tags/3.5.1
	mkdir build
	cd build
	#export SLEEF_ROOT=${PARSIM_ROOT}/sleef/install
	cmake -DCMAKE_INSTALL_PREFIX=${SLEEF_ROOT} -DCMAKE_C_COMPILER=clang -DBUILD_TESTS=FALSE -DSLEEF_ENABLE_LLVM_BITCODE=TRUE ..
	make -j 16
	make install
	# Handling environments where sleef installs libs in ${SLEEF_ROOT}/lib instead of ${SLEEF_ROOT}/lib64
	if [ ! -d "${SLEEF_ROOT}/lib64/" ]; 
	then
		ln -s ${SLEEF_ROOT}/lib ${SLEEF_ROOT}/lib64
	fi
	cd ${PARSIM_ROOT}
fi

echo " "
echo "----------------- DONE ------------------"
