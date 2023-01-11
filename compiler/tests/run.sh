#!/bin/bash

set -e

PSV_EXTRA=""
FILES=""

while test $# -gt 0
do
    case "$1" in
        -E) PSV_EXTRA="${PSV_EXTRA} -Werror"
            ;;
        -I) PSV_EXTRA="${PSV_EXTRA} -Iwarnset"
            ;;
        *) 	FILES="${FILES} $1"
            ;;
    esac
    shift
done

if [ "$FILES" = "" ];
then
    FILES=`ls *.cpp`
fi

mkdir -p bin

for i in $FILES; do
  echo $i
  BIN=$(echo $i | sed "s/.cpp$//")
  cmd="parsimony -O3 -march=native -mprefer-vector-width=512 -I../../apps/synet-simd/src $i -o bin/$BIN --Xpsv=\"$PSV_EXTRA\" --Xtmp tmp"
  echo $cmd
  eval $cmd
  ./bin/$BIN
  echo
done
