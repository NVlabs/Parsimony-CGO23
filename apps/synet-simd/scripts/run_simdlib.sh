#!/bin/bash

# you can use -nt=1 instead of -mt=... (nt = number of iterations, mt = minimum time) to 
# just execute 1 iteration of each bench in case you only care about verifying correctness
#
# Also, use something like:

# /home/scratch.sysarch_nvresearch/bin/sde-external-8.69.1-2021-07-18-lin/sde64 -mix -- ../build/Test -nt=1 -fi=AbsDifferenceSum -lc=1"

# to obtain statistics of the executed basic blocks (very useful when trying to understand differences between our generated code and other approaches)
# use -top_blocks 100 or higher if expected basic blocks are missing
root_dir=${PARSIM_ROOT}/apps/synet-simd
CMD="${root_dir}/build/Test -mt=400 -wt=1 -lc=1 \
    -ot=psim_results.txt \
    -fi=AbsDifference \
    -fi=AbsDifferenceSum \
    -fi=AbsGradientSaturatedSum \
    -fi=AddFeatureDifference \
    -fi=AlphaBlending \
    -fi=AlphaFilling \
    -fi=NeuralConvert \
    -fi=AlphaPremultiply \
    -fi=AlphaUnpremultiply \
    -fi=BackgroundGrowRangeSlow \
    -fi=BackgroundGrowRangeFast \
    -fi=BackgroundIncrementCount \
    -fi=BackgroundAdjustRange \
    -fi=BackgroundShiftRange \
    -fi=BackgroundInitMask \
    -fi=Base64Decode \
    -fi=Base64Encode \
    -fi=BayerToBgr \
    -fi=BayerToBgra \
    -fi=BgrToBayer \
    -fi=Binarization \
    -fi=Conditional \
    -fe=HistogramConditional \
    -fe=EdgeBackground \
    -fe=ConditionalSquare \
    -fe=AveragingBinarizationV2 
"
echo $CMD "$@"
eval $CMD "$@"
