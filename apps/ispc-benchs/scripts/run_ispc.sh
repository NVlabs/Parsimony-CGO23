#!/bin/bash
root_dir=$PARSIM_ROOT/apps/ispc-benchs
build_dir=${root_dir}/build

if [ ! "${1}" == "aobench" ] && [ ! "${1}" == "mandelbrot" ] && [ ! "${1}" == "options" ] && [ ! "${1}" == "volume_rendering" ] && [ ! "${1}" == "noise" ]&& [ ! "${1}" == "stencil" ]&& [ ! "${1}" == "all" ]; then
	echo "Please provide ispc-bench or 'all'; one of [all, aobench, mandelbrot, options, volume_rendering, noise, stencil]"
	echo "For example: ./scripts/run_ispc.sh aobench"
	exit
fi

if [ "${1}" == "all" ]; then
	benchs='aobench stencil volume_rendering options mandelbrot noise'
else
	benchs=${1}
fi
echo "Benchmark Name, serial runtime (in million cycles), ispc runtime (in million cycles), psv runtime (in million cycles)"
for bench in ${benchs}
do
	source_dir=${root_dir}/ispc/${bench}
    cd ${build_dir}/${bench}
	if [ "${bench}" == "aobench" ]; then
		args=" 512 512 5 0 5"
	elif [ "${bench}" == "volume_rendering" ]; then
		args=" ${source_dir}/camera.dat ${source_dir}/density_highres.vol 5 0 5"
	else
		args=""
	fi
	
	./${bench} ${args}
done

