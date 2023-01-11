#!/bin/bash
root_dir=${PARSIM_ROOT}/apps/ispc-benchs
build_dir=${root_dir}/build

if [ "${1}" == "clean" ]; then
	make clean -C ${root_dir}/psv
	if [ -d ${build_dir} ] ; then
		rm -r ${build_dir}
	fi
	exit
fi

mkdir -p ${build_dir}
cd ${build_dir}
make -j 16 -C ${root_dir}/psv
CC=clang CXX=clang++ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ${root_dir}/ispc
make -j 16

echo "DONE"

