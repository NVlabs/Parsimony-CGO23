# Parsimony Compiler Documentation

The Parsimony prototype compiler is implemented using LLVM 15.0.1. This document is provided as a starting point for extending Parsimony and/or porting more benchmarks to Parsimony enabled C++.

## Using the Parsimony Compiler 

After installing Parsimony, run `${PARSIM_INSTALL_PATH}/bin/parsimony --help` for Parsimony compiler's usage information. 

To compile an example source code `${PARSIM_ROOT}/compiler/tests/simple.cpp` with LLVM -O3 optimizations enabled:
```
${PARSIM_INSTALL_PATH}/bin/parsimony -O3 ${PARSIM_ROOT}/compiler/tests/simple.cpp -o simple
```
This will generate a Parsimony vectorized binary `simple` in your working directory. 

Note that Parsimony creates a folder `tmp/` by default in your working directory with temporary files generated during Parsimony's compilation flow. Use `--Xtmp <path_to_tmp_dir>` argument to `parsimony` to specify a different folder for temporary files.

Additionally, any clang-compatible arguments (such as `-O3`) passed to `parsimony` will be carried over to all invocations of `clang++` within Parsimony's compilation flow.

## Parsimony Compilation Flow

The Parsimony compilation flow is defined in the routine `run_compiler_steps` in  `${PARSIM_ROOT}/compiler/parsimony.py`. The steps performed to compile source code written in Parsimony enabled C++ down to Parsimony vectorized x86 assembly are listed as follows.

1. Front-end: Parsimony's SPMD constructs are compiled down to LLVM IR by piggybacking on Clang support for the extraction of `#pragma omp parallel` code regions. Parsimony's front-end replaces `#psim` constructs with `#pragma omp parallel for`, runs Clang's preprocessor (`clang++ -E`), and compiles the preprocessor output to LLVM middle-end IR with autovectorization disabled (`-fno-vectorize -fno-slp-vectorize`). Please look at Section 4.1 of our CGO23 paper for more information on this step.

2. Middle-End Vectorization Pass: Calls `${PARSIM_INSTALL_PATH}/bin/psv` to vectorize the LLVM bitcode file obtained from the previous step. `FunctionVectorizer::vectorize()` in `{PARSIM_ROOT}/compiler/src/function.cpp` defines the middle-end vectorization steps and Section 4.2 of our CGO23 paper explains Parsimony's middle-end vectorizer in detail.
 
3. Back-End: Parsimony uses the default LLVM backend to generate an object file or binary containing Parsimony vectorized x86 assembly and links it with the Sleef vectorized math library.

## Parsimony API

As mentioned in our CGO23 paper, use `#psim gang_size(N)` to demarcate explicit SPMD parallel regions. The gang_size does not have to match the hardware's SIMD width but it has to be known at compile time. Use either `num_spmd_threads(M)` or `num_spmd_gangs(M)` along with the `#psim gang_size(N)` construct to specify the number of total threads or gangs respectively. Please look at Section 3 of our CGO23 paper for more information on Parsimony's programming model.

`${PARSIM_ROOT}/compiler/include/parsim.h` includes the provided Parsimony abstractions. We describe these Parsimony abstractions below.

### Parsimony thread indexing operations
#### `unsigned psim_get_lane_num()`: 
Returns the unique lane number within the SPMD gang.

#### `uint64_t psim_get_gang_num()`: 
Returns the gang number.

#### `unsigned psim_get_gang_size()`: 
Returns the compile-time fixed gang size.

#### `uint64_t psim_get_num_threads()`: 
Returns the total number of SPMD threads.

#### `uint64_t psim_get_thread_num()`: 
Returns the unique SPMD thread number within the SPMD region.

### Identification of head and tail gang
Section 3 of our CGO23 paper details the optimizations that are enabled by these two abstractions below.

#### `bool psim_is_tail_gang()`: 
Returns true when called by Parsimony threads in the last gang of the SPMD region.

#### `bool psim_is_head_gang()`: 
Returns true when called by Parsimony threads in the first gang of the SPMD region.

### Saturating math operations
Parsimony's saturating math operations generate the corresponding LLVM [Saturation Arithmetic Intrinsic](https://llvm.org/docs/LangRef.html#saturation-arithmetic-intrinsics) in the middle-end IR.

#### `T psim_sadd_sat(T a, T b)`: 

Signed saturated addition of `a` and `b`. The arguments `a` and `b` and the result may be of integer types of any bit width, but they must have the same bit width. The maximum value this operation can clamp to is the largest signed value representable by the bit width of `a` and `b`. The minimum value is the smallest signed value representable by this bit width. 

#### `T psim_uadd_sat(T a, T b)`: 

Unsigned saturated addition of `a` and `b`. The arguments `a` and `b` and the result may be of integer types of any bit width, but they must have the same bit width. The maximum value this operation can clamp to is the largest unsigned value representable by the bit width of `a` and `b`. The result will never saturate towards zero because this is an unsigned operation.

#### `T psim_ssub_sat(T a, T b)`: 

Signed saturated subtraction of of `a` and `b`. The arguments `a` and `b` and the result may be of integer types of any bit width, but they must have the same bit width. The maximum value this operation can clamp to is the largest signed value representable by the bit width of `a` and `b`. The minimum value is the smallest signed value representable by this bit width. 

#### `T psim_usub_sat(T a, T b)`: 

Unsigned saturated subtraction of of `a` and `b`. The arguments `a` and `b` and the result may be of integer types of any bit width, but they must have the same bit width. The maximum value this operation can clamp to is the largest unsigned value representable by the bit width of `a` and `b`. The result will never saturate towards zero because this is an unsigned operation.

### AVX-512 Specific Intrinsics
Due to lack of general-purpose compiler IR constructs for the two operations below, we use AVX-512 specific IR constructs for them. Please see Section 7 of our CGO23 paper for more information on these.

#### `uint16_t psim_umulh(uint16_t a, uint16_t b)`: 

Multiplies two unsigned 16-bit integers `a` and `b` and outputs the high 16 bits of the multiplication result. 

This generates the AVX-512 LLVM intrinsic `x86_avx512_pmulhu_w_512` that corresponds to AVX-512 intrinsic [`_mm512_mulhi_epu16`](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mulhi_epu16&ig_expand=5026). 

#### `PsimCollectiveAddAbsDiff`:

This Parsimony abstraction is used to efficiently accumulate a sum of absolute differences of two 8-bit values across all Parsimony threads.

This abstraction generates the AVX-512 LLVM intrinsic `x86_avx512_psad_bw_512` that corresponds to the AVX-512 intrinsic [`_mm512_sad_epu8`](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=5026,6051&text=mm512_sad). 

To use this abstraction: 
First, declare an opaque structure `_sum` outside the SPMD region with 
```
PsimCollectiveAddAbsDiff<uint64_t> _sum;
```

Then, perform the sum of absolute differences of two `uint8_t` values `a` and `b`  inside the SPMD region with 
````
_sum.AddAbsDiff(a,b);
````

Finally, perform the reduction/accumulation of values in this opaque structure `_sum` outside the SPMD region and store it into a `uint64_t` variable `result` with 
```
uint64_t result = _sum.ReduceSum();
```

`${PARSIM_ROOT}/compiler/tests/sad.cpp` shows an example use of `PsimCollectiveAddAbsDiff`.
Note: This abstraction can also be used to perform an efficent sum reduction by passing `0` as the second argument of `_sum.AddAbsDiff(a,b)` in the example above. See routine `ConditionalCount8u` in `${PARSIM_ROOT}/apps/synet-simd/psimdlib/Conditional.cpp` for an example use case of `PsimCollectiveAddAbsDiff` to perform sum reduction.


### Horizontal shuffle and data exchange operations

All threads within a gang must participate in these operations. These operations explicitly synchronize across threads within a gang.

#### `T1 psim_shuffle_sync<T1>(T2 a, int src_lane)`: 

`psim_shuffle_sync` allows exchanging of a variable between Parsimony threads within a gang (also called lanes). The exchange occurs simultaneously for all lanes within the gang, copying variable `a` from indexed lane `src_lane`. The value of `src_lane` must be known at compile-time because LLVM's [`shufflevector`](https://llvm.org/docs/LangRef.html#shufflevector-instruction) IR instruction does not support shuffle indices not known at compile time. `src_lane` can be dependent on `psim_get_lane_num()` or other compile-time known constants per thread. Thus, `psim_shuffle_sync` returns the value of `a` held by lane `src_lane`. 

Note, if `src_lane` equates to `< 0` or `>= psim_get_gang_size()` for a Parsimony thread, then the value returned from `psim_shuffle_sync` for that thread is `0`. `T1` and `T2` can be of different types. `${PARSIM_ROOT}/compiler/tests/shuffle.cpp` shows some example uses of `psim_shuffle_sync`. 


#### `T1 psim_shuffle_sync<T1>(T2 a, T2 b, int src_index)`:
This is a two input version of the `psim_shuffle_sync` operation explained above and generates LLVM's [`shufflevector`](https://llvm.org/docs/LangRef.html#shufflevector-instruction) IR instruction.

In this case, the vector containing values for variable `a` across all lanes within a gang is concatenated with the vector containing values for `b` across all lanes within a gang, and `src_index` points to an index within this concatenated vector. Thus `src_index` must be `0 <= src_index < psim_get_gang_size()*2`, otherwise the returned value will be `0`. `a` and `b` must have the same type `T2` and the return type `T1` can be different from `T2`.

#### `T1 psim_zip_sync<T1>(T2 a)`:

`psim_zip_sync` 'zips' together the variable `a` across every `sizeof(T1)/sizeof(T2)` lanes within a gang and stores them in the first `psim_get_gang_size()*sizeof(T2)/sizeof(T1)` lanes. For example, when gang size is 32, `T1` is `uint32_t` and `T2` is `uint8_t`:  `uint32_t b = psim_zip_sync<uint32_t>((uint8_t)a)` will concatenate the 8-bit variable `a` from input lanes 0,1,2,3 together to store into 32-bit variable `b` at output lane 0. Similarly, the 32-bit variable `b` at lane 1 will have the concatenation of variable `a` from input lanes 4,5,6,7 and the 32-bit variable `b` at lane 7 will have the concatenation of 8-bit variable `a` from input lanes 28,29,30,31. The bitwidth of `T1` must be a multiple of the bitwidth of `T2`. Additionally, gang size must be divisible by the factor `sizeof(T1)/sizeof(T2)`. 

See `${PARSIM_ROOT}/compiler/tests/zip_unzip.cpp` for an example use of `psim_zip_sync`. 

#### `T1 psim_unzip_sync<T1>(T2 a, uint32_t index)`:

`psim_unzip_sync` performs the opposite data exchange operation of `psim_zip_sync`. It 'unzips' the variable `a` of type `T2` from a lane within a gang and disperses the outputs of type `T1` to `sizeof(T2)/sizeof(T1)` lanes within the same gang. `index` specifies which set of `psim_get_gang_size()*sizeof(T1)/sizeof(T2)` lanes within the gang to use as input to `psim_unzip_sync`. `index` must be between `0 <= index < sizeof(T2)/sizeof(T1)`. `T2` must be a multiple of `T1`. Additionally, gang size must be divisible by the factor `sizeof(T2)/sizeof(T1)`. 

For example, when gang size is 32, `T1` is `uint8_t` and `T2` is `uint32_t`:  `uint8_t b = psim_unzip_sync<uint8_t>((uint32_t)a, 0)` will divide the 32-bit variable `a` of input lane 0 into 4 8-bit values and disperses these 8-bit values to variable `b` at output lanes 0,1,2,3 respectively. `b` at lane 0 will get the most significant 8 bits of the 32-bit input `a` from lane 0 and `b` and lane 3 will get the least significant 8 bits of input `a` from lane 0. If index passed into `psim_unzip_sync` for this example was 2 instead of 0, then the same operations will take place for output variable `b` at lanes 0,1,2,3 except that their corresponding input variable `a` will now be from lane 16 instead of lane 0. Thus, within each gang, the input variable `a` is used from lanes lanes 0 to 7 when index=0, lanes 8 to 15 when index=1, lanes 16 to 23 when index=2, and lanes 24 to 31 when index=3.   

See `${PARSIM_ROOT}/compiler/tests/zip_unzip.cpp` for an example use of `psim_unzip_sync`. 
### Horizontal synchronization

#### `void psim_gang_sync()`: 
Provides explicit synchronization across Parsimony threads within the gang. 

### Atomic Sum Reduction

#### `void psim_atomic_add_local(T1* a, T2 value)`: 

With `a` being a pointer declared outside the SPMD region, `psim_atomic_add_local` will atomically perform the horizontal sum of variable `value` across all Parismony threads and store the result at the memory location pointed to by `a`. 

This abstraction generates the LLVM vector reduction intrinsic [llvm.vector.reduce.add](https://llvm.org/docs/LangRef.html#llvm-vector-reduce-add-intrinsic) or [llvm.vector.reduce.fadd](https://llvm.org/docs/LangRef.html#llvm-vector-reduce-fadd-intrinsic) depending on the datatype of `T2`.

## Extending the Parsimony API set

As an example, listed below are the steps to extend Parsimony's API with a atomic multiply reduction that performs a horizontal multiplication of a variable across all Parsimony threads and stores the final result at a memory location given by the user. Similar to `psim_atomic_add_local`, we can implement this by leveraging LLVM's vector reduction intrinsic [llvm.vector.reduce.mul](https://llvm.org/docs/LangRef.html#llvm-vector-reduce-mul-intrinsic) or [llvm.vector.reduce.fmul](https://llvm.org/docs/LangRef.html#llvm-vector-reduce-fmul-intrinsic) depending on the datatype of the input variable.

1. Declare the new API in `${PARSIM_ROOT}/compiler/include/parsim.h`, something like:
```
template <typename T1, typename T2>
void psim_atomic_mul_local(T1* a, T2 value) noexcept;
```

2. In `{PARSIM_ROOT}/compiler/src/resolver.h`, add an entry `ATOMICMUL_LOCAL` to `PsimApiEnum` and `{ATOMICMUL_LOCAL, "psim_atomic_mul_local"}`  to `PsimApiEnumStrMap`.


3. In `{PARSIM_ROOT}/compiler/src/shapes.cpp`, specify the shape of the new `FunctionResolver::PsimApiEnum::ATOMICMUL_LOCAL` at `Shape ShapesStep::calculateShapeCall(CallInst* call)` routine. Similar to `ATOMICADD_LOCAL`, we can assign `ATOMICMUL_LOCAL`'s shape to be `Shape::None()`.

4. In `{PARSIM_ROOT}/compiler/src/transform.cpp`, similar to the transformation for `ATOMICADD_LOCAL`, specify the transformation for `ATOMICMUL_LOCAL` to generate the appropriate LLVM vector reduction intrinsic in the `Value* TransformStep::transformCallPsimApi(llvm::CallInst* inst)` rotuine.
