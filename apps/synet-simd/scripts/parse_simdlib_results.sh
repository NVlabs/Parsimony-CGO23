#!/bin/bash

if [ "${1}" == "" ]; then
	echo "Please provide psim_results file."
	echo "For example: ./psimdlib/parse_simdlib_results.sh psim_results.txt"
	exit
fi

results_file=${1}
num_lines=`wc -l < ${results_file}`
num_benchs=`expr $num_lines - 9`

echo "Benchmark Name, scalar runtime (in ms), autovec runtime (in ms), avx512 runtime (in ms), parsimony runtime (in ms)"
tail -n ${num_benchs}  ${results_file} | head -n -1 | awk -F',' '{print $1,",",$3,",",$4,",",$8,",",$9}' | cut -c3- | sed -r 's/\s+//g'
