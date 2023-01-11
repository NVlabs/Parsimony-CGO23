#!/bin/bash
root_dir=${PARSIM_ROOT}/apps/synet-simd
build_dir=${root_dir}/build

if [ "${1}" == "clean" ]; then
	make clean -C ${root_dir}/psimdlib
	if [ -d ${build_dir} ] ; then
		rm -r ${build_dir}
	fi
	exit
fi

mkdir -p ${build_dir}
cd ${build_dir}
make -j 16 -C ${root_dir}/psimdlib
cmake ../prj/cmake -DSIMD_TOOLCHAIN="clang++" -DSIMD_TARGET=""
make -j 16

echo "DONE"

