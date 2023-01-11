# Parsimony CGO'23 Artifact Appendix Manual

Parsimony is a SPMD programming approach built with semantics designed to be compatible with multiple languages and to cleanly integrate into the standard optimizing compiler toolchains for those languages. If you use any component of Parsimony, please cite:
```
Vijay Kandiah, Daniel Lustig, Oreste Villa, David Nellans, Nikos Hardavellas,
Parsimony: Enabling SIMD/Vector Programming in Standard Compiler Flows,
in 2023 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)
```
This Repository serves as an artifact for the paper above and includes a LLVM prototype of the Parsimony model along with a build and test framework for the Simd Library benchmarks and ispc benchmarks. It also includes scripts to build our prototype compiler, build and run the SimdLibrary and ispc benchmarks, and reproduce the figures 4 and 5 presented in our CGO'23 paper.

## Prerequisites

* Our Parsimony prototype needs to be built and run on a machine with AVX-512, specifically `avx512bw`, i.e., `lscpu | grep -c avx512bw` should return 1.
* [LLVM 15.0.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-15.0.1) ([Additional dependencies](https://llvm.org/docs/GettingStarted.html#requirements))
* [ispc v1.18.0](https://github.com/ispc/ispc/releases/tag/v1.18.0) (Required for compiling ispc benchmarks)
* [Z3Prover](https://github.com/Z3Prover/z3)
* [Sleef](https://sleef.org/compile.xhtml) (Building Sleef requires `cmake --version` >= 3.5.1.)

## Authors' Environment Specifications
We performed all experiements in this environment:
* CPU: Intel(R) Xeon(R) Gold 6258R CPU
* Operating System: Red Hat Enterprise Linux Version 8.7 (Ootpa)
* `gcc`: v8.5.0
* `clang`: v15.0.1
* `cmake`: v3.20.2
* `python`: v2.7.18
* `python3`: v3.6.8
* `perl`: v5.26.3

## Wrapper Scripts
We provide a set of wrapper scripts in `<artifact_repo_root>/scripts` to do all the steps listed below this section. 
Before running any of these scripts, please set the environment variable `PARSIM_ROOT` to the root of this repository with:
```
export PARSIM_ROOT=<artifact_repo_root>
```

After setting `PARSIM_ROOT`, please set up the environment with:
```
source $PARSIM_ROOT/scripts/setup_environment.sh
``` 
This exports the necessary paths for the scripts below to work. Additionally, it adds ispc and Parsimony binaries to your `${PATH}`. This step is not to be used if you are following the detailed instructions below this section because the location at which you install prereqs and Parsimony might be different.

### To install LLVM 15.0.1:

Install LLVM 15.0.1 manually with: 
```
git clone --depth=1 --single-branch --branch=llvmorg-15.0.1 https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS='clang' -DLLVM_TARGETS_TO_BUILD="AArch64;X86" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<llvm-install-path> -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;openmp" ../llvm
cmake --build . -j 16 --target install
export PATH=<llvm-install-path>/bin:${PATH}
export LD_LIBRARY_PATH=<llvm-install-path>/lib:${LD_LIBRARY_PATH}
export LLVM_ROOT=<llvm-install-path>
```
This might require installation of additional dependencies listed in LLVM's [GettingStarted](https://llvm.org/docs/GettingStarted.html#requirements) page. 

Alternatively, install LLVM through your preferred package system:
```
sudo apt install clang-15 --install-suggests
#or
sudo yum install llvm-toolset-15.0.1

export PATH=<llvm-install-path>/bin:${PATH}
export LD_LIBRARY_PATH=<llvm-install-path>/lib:${LD_LIBRARY_PATH}
export LLVM_ROOT=<llvm-install-path>
```

### To install all other prerequisites: 
```
$PARSIM_ROOT/scripts/install_prereqs.sh
```
Note that the above command uses 16 cores (-j 16) for some of the build stages. This command took about 20 minutes to complete on our machine.


### To install the Parsimony compiler, Simd Library benchmarks, and ispc Benchmarks:
```
$PARSIM_ROOT/scripts/install_psim_and_apps.sh
```
Note that the above command uses 16 cores (-j 16) for some of the build stages. This command took about 5 minutes to complete on our machine.


### To run all Simd Library and ispc benchmarks:
```
$PARSIM_ROOT/scripts/run_apps.sh
```
This command runs the benchmarks one after the other on a single core and takes about 20 minutes to complete. Please make sure your system is idle while running this command to minimize noise in the collected performance results.
This should write the performance results of Simd Library and ispc benchmarks to `$PARSIM_ROOT/results/simdlib_results.csv` and `$PARSIM_ROOT/results/ispcbench_results.csv` respectively. If you have reached this point, please proceed to the section below about [generating Figures 4 and 5](#generating-figures-4-and-5-in-our-cgo23-paper) 

### To uninstall the Parsimony compiler and all Simd Library and ispc benchmarks:
```
$PARSIM_ROOT/scripts/uninstall_psim_and_apps.sh
```
Detailed instructions can be found in the sections below.

***

## Building Prerequisites

### LLVM
  
Build LLVM 15.0.1 manually with: 
```
git clone --depth=1 --single-branch --branch=llvmorg-15.0.1 https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS='clang' -DLLVM_TARGETS_TO_BUILD="AArch64;X86" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<llvm-install-path> -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;openmp" ../llvm
cmake --build . -j 16 --target install
export PATH=<llvm-install-path>/bin:${PATH}
export LD_LIBRARY_PATH=<llvm-install-path>/lib:${LD_LIBRARY_PATH}
export LLVM_ROOT=<llvm-install-path>
```
This might require installation of additional dependencies listed in LLVM's [GettingStarted](https://llvm.org/docs/GettingStarted.html) documentation. 

Alternatively, install LLVM through your preferred package system:
```
sudo apt install clang-15 --install-suggests
#or
sudo yum install llvm-toolset-15.0.1

export PATH=<llvm-install-path>/bin:${PATH}
export LD_LIBRARY_PATH=<llvm-install-path>/lib:${LD_LIBRARY_PATH}
export LLVM_ROOT=<llvm-install-path>
```

### ispc 
Please use the ispc v1.18.0 pre-built binary:
```
wget https://github.com/ispc/ispc/releases/download/v1.18.0/ispc-v1.18.0-linux.tar.gz
tar -xvf ispc-v1.18.0-linux.tar.gz
export PATH=<path-to-working-dir>/ispc-v1.18.0-linux/bin:${PATH}
```
Alternatively, building ispc v1.18.0 from source code requires LLVM 13.0.0. Please follow ispc's [documentation](https://github.com/ispc/ispc/wiki/ISPC-Development-Guide) to do so. 

### Z3Prover
To build Z3:
```
git clone https://github.com/Z3Prover/z3.git
cd z3
git checkout tags/z3-4.11.2
CXX=clang++ CC=clang python scripts/mk_make.py --prefix=<z3-install-path>
cd build
make -j 16
make install
``` 

### Building Sleef
To build Sleef:
```
git clone https://github.com/shibatch/sleef.git
cd sleef
git checkout tags/3.5.1
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=<sleef-install-path> -DCMAKE_C_COMPILER=clang -DBUILD_TESTS=FALSE -DSLEEF_ENABLE_LLVM_BITCODE=TRUE ..
make -j 16
make install
```
Note: Depending on your environment, Sleef might install libraries in `${SLEEF_ROOT}/lib` instead of the expected `${SLEEF_ROOT}/lib64`. Hence use the following command to create a symbolic link if necessary.
```
if [ ! -d "${SLEEF_ROOT}/lib64/" ];  then ln -s ${SLEEF_ROOT}/lib ${SLEEF_ROOT}/lib64; fi
```

## Building Parsimony

After installing the prerequisites, build our Parsimony compiler with:  
```
export PARSIM_ROOT=<artifact_repo_root>
export LLVM_ROOT=<llvm-install-path>
export Z3_ROOT=<z3-install-path>
export SLEEF_ROOT=<sleef-install-path>

export PARSIM_INSTALL_PATH=$PARSIM_ROOT/compiler/install  # or something else if you prefer
cd $PARSIM_ROOT/compiler
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_INSTALL_DIR=$LLVM_ROOT -DSLEEF_INSTALL_DIR=$SLEEF_ROOT -DCMAKE_INSTALL_PREFIX=$PARSIM_INSTALL_PATH -DZ3_INSTALL_DIR=$Z3_ROOT ..
cmake --build . -j 16 --target install
export PATH=$PARSIM_INSTALL_PATH/bin:$PATH
```

The Sleef vector math-library is optional, omit `-DSLEEF_INSTALL_DIR` variable in the cmake command below if sleef is not required. However, note that we require it to generate the same results presented in Figure 4 of our CGO'23 paper.

It is also possible to optinally specify a different path for the final LLVM backend (the one that compiles the post-vectorized file into the final object) by using:
```
export LLVM_BACKEND_ROOT=<llvm_backend_install_path>
-DLLVM_BACKEND_DIR=$LLVM_BACKEND_ROOT
```
This is particularly important in order to get the latest bug fixes done to LLVM's final code generator, without needing to re-base our compiler pass to the latest version of LLVM.


## Simple Parsimony Tests
There are a number of simple Parsimony tests provided at `$PARSIM_ROOT/compiler/tests`.
To build and run them:
```
cd $PARSIM_ROOT/compiler/tests
./run.sh [cpp_file]
```

## Parsimony Compiler Documentation
The `$PARSIM_ROOT/compiler/README.md` file contains documentation about using the Parsimony compiler, the Parsimony compilation flow, the provided Parsimony API feature set, and steps for extending the Parsimony API set. This file is provided as a starting point for extending Parsimony and/or porting more benchmarks to Parsimony enabled C++.


## The Simd Library
`$PARSIM_ROOT/apps/synet-simd` contains our fork of the [Simd Library](https://github.com/ermig1979/Simd).
`$PARSIM_ROOT/apps/synet-simd/psimdlib` contains Parsimony implementations of Simd Library benchmarks.

### Building The Simd Library

To build our fork of the [Simd Library](https://github.com/ermig1979/Simd):

```
$PARSIM_ROOT/apps/synet-simd/scripts/build_simdlib.sh
```

To clean/uninstall The Simd Library build
```
$PARSIM_ROOT/apps/synet-simd/scripts/build_simdlib.sh clean
```


### Running The Simd Library Benchmarks

```
$PARSIM_ROOT/apps/synet-simd/scripts/run_simdlib.sh
```

Should result in something like:

```
$PARSIM_ROOT/apps/synet-simd/build/Test -mt=400 -wt=1 -lc=1 -ot=psim_results.txt -fi=AbsDifference -fi=AbsDifferenceSum -fi=AbsGradientSaturatedSum -fi=AddFeatureDifference -fi=AlphaBlending -fi=AlphaFilling -fi=NeuralConvert -fi=AlphaPremultiply -fi=AlphaUnpremultiply -fi=BackgroundGrowRangeSlow -fi=BackgroundGrowRangeFast -fi=BackgroundIncrementCount -fi=BackgroundAdjustRange -fi=BackgroundShiftRange -fi=BackgroundInitMask -fi=Base64Decode -fi=Base64Encode -fi=BayerToBgr -fi=BayerToBgra -fi=BgrToBayer -fi=Binarization -fi=Conditional -fe=HistogramConditional -fe=EdgeBackground -fe=ConditionalSquare -fe=AveragingBinarizationV2
[000] Info: AbsDifferenceAutoTest is started :
[000] Info: Test Simd::Base::AbsDifference & SimdAbsDifference [1920, 1080].
[000] Info: Test Simd::Base::AbsDifference & SimdAbsDifference [1929, 1071].

[...]

Simd Library Performance Report:

Test generation time: 2022.11.14 16:32:09

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Function                     |    API  Scalar AutoVec   Sse2  Sse41   Avx2  Avx5b    Psv | Scal/AVec Scal/S2 Scal/S4 Scal/A2 Scal/A6 Scal/Ps | Scal/AVec AVec/S2 S2/S4 S4/A2 A2/A6 A6/Ps |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Common, ms,                  | 1.198,  6.095,  2.280, 1.423, 1.373, 1.280, 1.176, 1.267, |     2.67,   4.28,   4.44,   4.76,   5.18,   4.81, |     2.67,   1.60, 1.04, 1.07, 1.09, 0.93, |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| AbsDifference,               | 1.804,  3.857,  2.106, 2.252,      , 1.205, 1.906, 2.005, |     1.83,   1.71,       ,   3.20,   2.02,   1.92, |     1.83,   0.94,     , 1.87, 0.63, 0.95, |
| AbsDifferenceSum,            | 0.692,  1.812,  1.003, 0.864,      , 0.758, 0.657, 0.790, |     1.81,   2.10,       ,   2.39,   2.76,   2.29, |     1.81,   1.16,     , 1.14, 1.15, 0.83, |

```

Additionally, a `psim_results.txt` file containing this table will be stored in the working directory. 
Please make sure your system is idle while running this command to minimize noise in the collected performance results.

### Parsing The Simd Library Results

To parse the results from the newly created `psim_results.txt` in the `.csv` format required for generating Figure 5 of our CGO'23 paper:
```
$PARSIM_ROOT/apps/synet-simd/scripts/parse_simdlib_results.sh <path-to-psim_results.txt> &> $PARSIM_ROOT/results/simdlib_results.csv
```


## ispc Benchmarks

`$PARSIM_ROOT/apps/ispc-benchs` contains our fork of the [ispc benchmarks repository](https://github.com/ermig1979/Simd).
`$PARSIM_ROOT/apps/ispc-benchs/psv` contains Parsimony implementations of these ispc benchmarks.

### Building ispc Benchmarks

To build our fork of the [ispc benchmarks repository](https://github.com/ermig1979/Simd):

```
$PARSIM_ROOT/apps/ispc-benchs/scripts/build_ispc.sh
```

To clean/uninstall The Simd Library build
```
$PARSIM_ROOT/apps/ispc-benchs/scripts/build_ispc.sh clean
```


### Running ispc Benchmarks

To run the Parsimony, ispc, and LLVM-15 auto-vectorized version of a specific ispc benchmark among {aobench, mandelbrot, options, volume_rendering, noise, stencil}:
```
$PARSIM_ROOT/apps/ispc-benchs/scripts/run_ispc.sh aobench

```

To run all ispc benchmarks for which we have Parsimony implementations:
```
$PARSIM_ROOT/apps/ispc-benchs/scripts/run_ispc.sh all > $PARSIM_ROOT/results/ispcbench_results.csv
```
Please make sure your system is idle while running this command to minimize noise in the collected performance results.
The command above will generate the ispc benchmark results in the `.csv` format required for generating Figure 4 of our CGO'23 paper.


## Generating Figures 4 and 5 in our CGO'23 paper

### Generating Figure 4

At this point you should have a CSV file `$PARSIM_ROOT/results/ispcbench_results.csv` containing runtime (in million cycles) for LLVM-15 autovectorized implementation, ispc implementation, and Parsimony implementation for the 7 ispc benchmarks.

The sheet `ispc-benchs` in the provided excel file `$PARSIM_ROOT/results/parsimony_cgo23.xlsx` contains a table in the format required for generating the Figure 4 graph presented in our CGO'23 paper. Paste all rows except the header row from `$PARSIM_ROOT/results/ispcbench_results.csv` into the cells `B2:E8` (highlighted in grey) in the Sheet `ispc-benchs` in `$PARSIM_ROOT/results/parsimony_cgo23.xlsx`. This should automatically generate the figure 4 graph which is located below the table in the same sheet. 

### Generating Figure 5

At this point you should have a CSV file `$PARSIM_ROOT/results/simdlib_results.csv` containing runtime (in ms) for LLVM-15 scalar implementation, LLVM-15 autovectorized implementation, AVX-512 Hand-written implementation, and Parsimony implementation for 72 Simd Library benchmarks.

The sheet `SimdLibrary` in the provided excel file `$PARSIM_ROOT/results/parsimony_cgo23.xlsx` contains a table in the format required for generating the Figure 5 graph presented in our CGO'23 paper. Paste all rows except the header row from `$PARSIM_ROOT/results/simdlib_results.csv` into the cells `B2:F73` (highlighted in grey) in the Sheet `SimdLibrary` in `$PARSIM_ROOT/results/parsimony_cgo23.xlsx`. This should automatically generate the figure 5 graph which is located below the table in the same sheet. 

Note that we sort the benchmarks by performance of Parsimony implementation in ascending order for Figure 5 in our paper. To do the same, you can sort the table by column header cell `H1` by clicking on `Sort Smallest to Largest` in the drop-down menu of this cell.

Additionally, a larger version of Figure 5 showing the Simd Library benchmark names is located further below on the same sheet. 


### Results from Authors' machine

The ispc benchmark and Simd Library results collected from our machine (Intel(R) Xeon(R) Gold 6258R) are provided in  `$PARSIM_ROOT/results/authors_results/` in both the `.csv` format and in the excel sheet `$PARSIM_ROOT/results/authors_results/parsimony_cgo23_authors.xlsx`
