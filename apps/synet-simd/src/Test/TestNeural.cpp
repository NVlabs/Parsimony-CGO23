/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "Test/TestUtils.h"
#include "Test/TestPerformance.h"
#include "Test/TestData.h"

#ifdef TEST_PERFORMANCE_TEST_ENABLE
#define SIMD_CHECK_PERFORMANCE() TEST_PERFORMANCE_TEST_(SIMD_FUNCTION)
#endif

#include "Simd/SimdNeural.hpp"
#include "psimdlib.h"
#include "Simd/SimdScalar.h"
namespace Test
{
    namespace
    {
        struct FuncC1
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride, int inversion);

            FuncPtr func;
            String description;
            bool inversion;

            FuncC1(const FuncPtr & f, const String & d, bool i) : func(f), description(d + (i ? "[1]" : "[0]")), inversion(i) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, (float*)dst.data, src.width, inversion ? 1 : 0);
            }
        };
    }
#define FUNC_C1(function, inversion) FuncC1(function, #function, inversion)

    bool NeuralConvertAutoTest(int width, int height, float eps, const FuncC1 & f1, const FuncC1 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        View dst1(width*height, 1, View::Float, NULL, TEST_ALIGN(width));
        View dst2(width*height, 1, View::Float, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = Compare(dst1, dst2, eps, true, 32);

        return result;
    }

    bool NeuralConvertAutoTest(float eps, const FuncC1 & f1, const FuncC1 & f2)
    {
        bool result = true;

        result = result && NeuralConvertAutoTest(W, H, eps, f1, f2);
        result = result && NeuralConvertAutoTest(W - O, H + O, eps, f1, f2);
        result = result && NeuralConvertAutoTest(W + O, H - O, eps, f1, f2);

        return result;
    }

    bool NeuralConvertAutoTest()
    {
        bool result = true;

        result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Base::NeuralConvert, true), FUNC_C1(SimdNeuralConvert, true));
        result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Base::NeuralConvert, false), FUNC_C1(SimdNeuralConvert, false));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
        {
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Sse2::NeuralConvert, true), FUNC_C1(SimdNeuralConvert, true));
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Sse2::NeuralConvert, false), FUNC_C1(SimdNeuralConvert, false));
        }
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
        {
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Avx2::NeuralConvert, true), FUNC_C1(SimdNeuralConvert, true));
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Avx2::NeuralConvert, false), FUNC_C1(SimdNeuralConvert, false));
        }
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
        {
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Avx512bw::NeuralConvert, true), FUNC_C1(SimdNeuralConvert, true));
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Avx512bw::NeuralConvert, false), FUNC_C1(SimdNeuralConvert, false));
        }
#endif

#ifdef SIMD_VSX_ENABLE
        if (Simd::Vsx::Enable)
        {
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Vsx::NeuralConvert, true), FUNC_C1(SimdNeuralConvert, true));
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Vsx::NeuralConvert, false), FUNC_C1(SimdNeuralConvert, false));
        }
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
        {
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Neon::NeuralConvert, true), FUNC_C1(SimdNeuralConvert, true));
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Neon::NeuralConvert, false), FUNC_C1(SimdNeuralConvert, false));
        }
#endif

        {
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Scalar::NeuralConvert, true), FUNC_C1(SimdNeuralConvert, true));
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Psv::NeuralConvert, true), FUNC_C1(SimdNeuralConvert, true));
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Scalar::NeuralConvert, false), FUNC_C1(SimdNeuralConvert, false));
            result = result && NeuralConvertAutoTest(EPS, FUNC_C1(Simd::Psv::NeuralConvert, false), FUNC_C1(SimdNeuralConvert, false));
        }
        return result;
    }

    namespace
    {
        struct FuncPS
        {
            typedef void(*FuncPtr)(const float * a, const float * b, size_t size, float * sum);

            FuncPtr func;
            String description;

            FuncPS(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & a, const View & b, float * sum) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)a.data, (float*)b.data, a.width, sum);
            }
        };
    }
#define FUNC_PS(function) FuncPS(function, #function)

    bool NeuralProductSumAutoTest(int size, float eps, const FuncPS & f1, const FuncPS & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View a(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(a);

        View b(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(b);

        float s1, s2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, &s1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, &s2));

        result = Compare(s1, s2, eps, true);

        return result;
    }

    bool NeuralProductSumAutoTest(float eps, const FuncPS & f1, const FuncPS & f2)
    {
        bool result = true;

        result = result && NeuralProductSumAutoTest(W*H, eps, f1, f2);
        result = result && NeuralProductSumAutoTest(W*H + O, eps, f1, f2);
        result = result && NeuralProductSumAutoTest(W*H - O, eps, f1, f2);

        return result;
    }

    bool NeuralProductSumAutoTest()
    {
        bool result = true;

        result = result && NeuralProductSumAutoTest(EPS, FUNC_PS(Simd::Base::NeuralProductSum), FUNC_PS(SimdNeuralProductSum));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralProductSumAutoTest(EPS, FUNC_PS(Simd::Sse2::NeuralProductSum), FUNC_PS(SimdNeuralProductSum));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralProductSumAutoTest(EPS, FUNC_PS(Simd::Avx::NeuralProductSum), FUNC_PS(SimdNeuralProductSum));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralProductSumAutoTest(EPS, FUNC_PS(Simd::Avx2::NeuralProductSum), FUNC_PS(SimdNeuralProductSum));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralProductSumAutoTest(EPS, FUNC_PS(Simd::Avx512f::NeuralProductSum), FUNC_PS(SimdNeuralProductSum));
#endif

#ifdef SIMD_VSX_ENABLE
        if (Simd::Vsx::Enable)
            result = result && NeuralProductSumAutoTest(EPS, FUNC_PS(Simd::Vsx::NeuralProductSum), FUNC_PS(SimdNeuralProductSum));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralProductSumAutoTest(EPS, FUNC_PS(Simd::Neon::NeuralProductSum), FUNC_PS(SimdNeuralProductSum));
#endif

        return result;
    }

    namespace
    {
        struct FuncAVMV
        {
            typedef void(*FuncPtr)(const float * src, size_t size, const float * value, float * dst);

            FuncPtr func;
            String description;

            FuncAVMV(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float value, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.width, &value, (float*)dstDst.data);
            }
        };
    }
#define FUNC_AVMV(function) FuncAVMV(function, #function)

    bool NeuralAddVectorMultipliedByValueAutoTest(int size, float eps, const FuncAVMV & f1, const FuncAVMV & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(src);

        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(dstSrc);

        const float value = 0.3f;

        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, value, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, value, dstSrc, dstDst2));

        result = Compare(dstDst1, dstDst2, eps, true);

        return result;
    }

    bool NeuralAddVectorMultipliedByValueAutoTest(float eps, const FuncAVMV & f1, const FuncAVMV & f2)
    {
        bool result = true;

        result = result && NeuralAddVectorMultipliedByValueAutoTest(W*H, eps, f1, f2);
        result = result && NeuralAddVectorMultipliedByValueAutoTest(W*H + O, eps, f1, f2);
        result = result && NeuralAddVectorMultipliedByValueAutoTest(W*H - O, eps, f1, f2);

        return result;
    }

    bool NeuralAddVectorMultipliedByValueAutoTest()
    {
        bool result = true;

        result = result && NeuralAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Base::NeuralAddVectorMultipliedByValue), FUNC_AVMV(SimdNeuralAddVectorMultipliedByValue));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Sse2::NeuralAddVectorMultipliedByValue), FUNC_AVMV(SimdNeuralAddVectorMultipliedByValue));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Avx::NeuralAddVectorMultipliedByValue), FUNC_AVMV(SimdNeuralAddVectorMultipliedByValue));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Avx2::NeuralAddVectorMultipliedByValue), FUNC_AVMV(SimdNeuralAddVectorMultipliedByValue));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Avx512f::NeuralAddVectorMultipliedByValue), FUNC_AVMV(SimdNeuralAddVectorMultipliedByValue));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Neon::NeuralAddVectorMultipliedByValue), FUNC_AVMV(SimdNeuralAddVectorMultipliedByValue));
#endif

        return result;
    }

    namespace
    {
        struct FuncAddVec
        {
            typedef void(*FuncPtr)(const float * src, size_t size, float * dst);

            FuncPtr func;
            String description;

            FuncAddVec(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.width, (float*)dstDst.data);
            }
        };
    }
#define FUNC_ADDVEC(function) FuncAddVec(function, #function)

    bool NeuralAddVectorAutoTest(int size, float eps, const FuncAddVec & f1, const FuncAddVec & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(src);

        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(dstSrc);

        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dstSrc, dstDst2));

        result = Compare(dstDst1, dstDst2, eps, true);

        return result;
    }

    bool NeuralAddVectorAutoTest(float eps, const FuncAddVec & f1, const FuncAddVec & f2)
    {
        bool result = true;

        result = result && NeuralAddVectorAutoTest(W*H, eps, f1, f2);
        result = result && NeuralAddVectorAutoTest(W*H + O, eps, f1, f2);
        result = result && NeuralAddVectorAutoTest(W*H - O, eps, f1, f2);

        return result;
    }

    bool NeuralAddVectorAutoTest()
    {
        bool result = true;

        result = result && NeuralAddVectorAutoTest(EPS, FUNC_ADDVEC(Simd::Base::NeuralAddVector), FUNC_ADDVEC(SimdNeuralAddVector));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddVectorAutoTest(EPS, FUNC_ADDVEC(Simd::Sse2::NeuralAddVector), FUNC_ADDVEC(SimdNeuralAddVector));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddVectorAutoTest(EPS, FUNC_ADDVEC(Simd::Avx::NeuralAddVector), FUNC_ADDVEC(SimdNeuralAddVector));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddVectorAutoTest(EPS, FUNC_ADDVEC(Simd::Avx512f::NeuralAddVector), FUNC_ADDVEC(SimdNeuralAddVector));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddVectorAutoTest(EPS, FUNC_ADDVEC(Simd::Neon::NeuralAddVector), FUNC_ADDVEC(SimdNeuralAddVector));
#endif

        return result;
    }

    namespace
    {
        struct FuncAddVal
        {
            typedef void(*FuncPtr)(const float * value, float * dst, size_t size);

            FuncPtr func;
            String description;

            FuncAddVal(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const float & value, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func(&value, (float*)dstDst.data, dstDst.width);
            }
        };
    }
#define FUNC_ADDVAL(function) FuncAddVal(function, #function)

    bool NeuralAddValueAutoTest(int size, float eps, const FuncAddVal & f1, const FuncAddVal & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(dstSrc);

        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        const float value = 3.14f;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(value, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(value, dstSrc, dstDst2));

        result = Compare(dstDst1, dstDst2, eps, true);

        return result;
    }

    bool NeuralAddValueAutoTest(float eps, const FuncAddVal & f1, const FuncAddVal & f2)
    {
        bool result = true;

        result = result && NeuralAddValueAutoTest(W*H, eps, f1, f2);
        result = result && NeuralAddValueAutoTest(W*H + O, eps, f1, f2);
        result = result && NeuralAddValueAutoTest(W*H - O, eps, f1, f2);

        return result;
    }

    bool NeuralAddValueAutoTest()
    {
        bool result = true;

        result = result && NeuralAddValueAutoTest(EPS, FUNC_ADDVAL(Simd::Base::NeuralAddValue), FUNC_ADDVAL(SimdNeuralAddValue));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddValueAutoTest(EPS, FUNC_ADDVAL(Simd::Sse2::NeuralAddValue), FUNC_ADDVAL(SimdNeuralAddValue));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddValueAutoTest(EPS, FUNC_ADDVAL(Simd::Avx::NeuralAddValue), FUNC_ADDVAL(SimdNeuralAddValue));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddValueAutoTest(EPS, FUNC_ADDVAL(Simd::Avx512f::NeuralAddValue), FUNC_ADDVAL(SimdNeuralAddValue));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddValueAutoTest(EPS, FUNC_ADDVAL(Simd::Neon::NeuralAddValue), FUNC_ADDVAL(SimdNeuralAddValue));
#endif

        return result;
    }

    namespace
    {
        struct FuncAF
        {
            typedef void(*FuncPtr)(const float * src, size_t size, const float * slope, float * dst);

            FuncPtr func;
            String description;

            FuncAF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float slope, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.width, &slope, (float*)dst.data);
            }
        };
    }
#define FUNC_AF(function) FuncAF(function, #function)

    bool NeuralActivateFunctionAutoTest(int size, float error, bool relative, float slope, float lo, float hi, const FuncAF & f1, const FuncAF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(src, lo, hi);

        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, slope, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, slope, dst2));

        result = Compare(dst1, dst2, error, true, 32, relative);

        return result;
    }

    bool NeuralActivateFunctionAutoTest(float error, bool relative, float slope, const FuncAF & f1, const FuncAF & f2)
    {
        bool result = true;

        const float lo = -10.0f, hi = 10.0f;
        result = result && NeuralActivateFunctionAutoTest(W*H, error, relative, slope, lo, hi, f1, f2);
        result = result && NeuralActivateFunctionAutoTest(W*H + O, error, relative, slope, lo, hi, f1, f2);

        return result;
    }

    bool NeuralRoughSigmoidAutoTest()
    {
        bool result = true;

        result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Base::NeuralRoughSigmoid), FUNC_AF(SimdNeuralRoughSigmoid));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Sse2::NeuralRoughSigmoid), FUNC_AF(SimdNeuralRoughSigmoid));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Avx::NeuralRoughSigmoid), FUNC_AF(SimdNeuralRoughSigmoid));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Avx512f::NeuralRoughSigmoid), FUNC_AF(SimdNeuralRoughSigmoid));
#endif

#ifdef SIMD_VSX_ENABLE
        if (Simd::Vsx::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Vsx::NeuralRoughSigmoid), FUNC_AF(SimdNeuralRoughSigmoid));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Neon::NeuralRoughSigmoid), FUNC_AF(SimdNeuralRoughSigmoid));
#endif

        return result;
    }

    bool NeuralRoughSigmoid2AutoTest()
    {
        bool result = true;

        result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Base::NeuralRoughSigmoid2), FUNC_AF(SimdNeuralRoughSigmoid2));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Sse2::NeuralRoughSigmoid2), FUNC_AF(SimdNeuralRoughSigmoid2));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Avx::NeuralRoughSigmoid2), FUNC_AF(SimdNeuralRoughSigmoid2));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Avx2::NeuralRoughSigmoid2), FUNC_AF(SimdNeuralRoughSigmoid2));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Avx512f::NeuralRoughSigmoid2), FUNC_AF(SimdNeuralRoughSigmoid2));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Neon::NeuralRoughSigmoid2), FUNC_AF(SimdNeuralRoughSigmoid2));
#endif

        return result;
    }

    bool NeuralRoughTanhAutoTest()
    {
        bool result = true;

        result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Base::NeuralRoughTanh), FUNC_AF(SimdNeuralRoughTanh));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Sse2::NeuralRoughTanh), FUNC_AF(SimdNeuralRoughTanh));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Avx::NeuralRoughTanh), FUNC_AF(SimdNeuralRoughTanh));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Avx512f::NeuralRoughTanh), FUNC_AF(SimdNeuralRoughTanh));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralActivateFunctionAutoTest(EPS, false, 1.1f, FUNC_AF(Simd::Neon::NeuralRoughTanh), FUNC_AF(SimdNeuralRoughTanh));
#endif

        return result;
    }

    bool NeuralPowAutoTest(float error, bool relative, const FuncAF & f1, const FuncAF & f2)
    {
        bool result = true;

#if defined(SIMD_NEON_ENABLE)
        const float lo = 0.002f, hi = 9.998f, exponent = -0.75;
#else
        const float lo = 0.001f, hi = 9.999f, exponent = -0.75;
#endif
        result = result && NeuralActivateFunctionAutoTest(W*H, error, relative, exponent, lo, hi, f1, f2);
        result = result && NeuralActivateFunctionAutoTest(W*H + O, error, relative, exponent, lo, hi, f1, f2);

        return result;
    }

    bool NeuralPowAutoTest()
    {
        bool result = true;

        result = result && NeuralPowAutoTest(EPS, false, FUNC_AF(Simd::Base::NeuralPow), FUNC_AF(SimdNeuralPow));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralPowAutoTest(EPS, false, FUNC_AF(Simd::Sse2::NeuralPow), FUNC_AF(SimdNeuralPow));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralPowAutoTest(EPS, false, FUNC_AF(Simd::Avx2::NeuralPow), FUNC_AF(SimdNeuralPow));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralPowAutoTest(EPS, false, FUNC_AF(Simd::Avx512f::NeuralPow), FUNC_AF(SimdNeuralPow));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralPowAutoTest(EPS, false, FUNC_AF(Simd::Neon::NeuralPow), FUNC_AF(SimdNeuralPow));
#endif 

        return result;
    }

    namespace
    {
        struct FuncAD
        {
            typedef void(*FuncPtr)(const float * src, size_t size, const float * slope, float * dst);

            FuncPtr func;
            String description;

            FuncAD(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float slope, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.width, &slope, (float*)dstDst.data);
            }
        };
    }
#define FUNC_AD(function) FuncAD(function, #function)

    bool NeuralActivateDerivativeAutoTest(int size, float error, bool relative, float slope, const FuncAD & f1, const FuncAD & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(src, -10.0f, 10.0f);

        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(dstSrc, -1.0f, 1.0f);

        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, slope, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, slope, dstSrc, dstDst2));

        result = Compare(dstDst1, dstDst2, error, true, 32, relative);

        return result;
    }

    bool NeuralActivateDerivativeAutoTest(float error, bool relative, float slope, const FuncAD & f1, const FuncAD & f2)
    {
        bool result = true;

        result = result && NeuralActivateDerivativeAutoTest(W*H, error, relative, slope, f1, f2);
        result = result && NeuralActivateDerivativeAutoTest(W*H + O, error, relative, slope, f1, f2);
        result = result && NeuralActivateDerivativeAutoTest(W*H - O, error, relative, slope, f1, f2);

        return result;
    }

    bool NeuralDerivativeSigmoidAutoTest()
    {
        bool result = true;

        result = result && NeuralActivateDerivativeAutoTest(EPS, true, 3.0f, FUNC_AD(Simd::Base::NeuralDerivativeSigmoid), FUNC_AD(SimdNeuralDerivativeSigmoid));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 3.0f, FUNC_AD(Simd::Sse2::NeuralDerivativeSigmoid), FUNC_AD(SimdNeuralDerivativeSigmoid));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 3.0f, FUNC_AD(Simd::Avx::NeuralDerivativeSigmoid), FUNC_AD(SimdNeuralDerivativeSigmoid));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 3.0f, FUNC_AD(Simd::Avx512f::NeuralDerivativeSigmoid), FUNC_AD(SimdNeuralDerivativeSigmoid));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 3.0f, FUNC_AD(Simd::Neon::NeuralDerivativeSigmoid), FUNC_AD(SimdNeuralDerivativeSigmoid));
#endif

        return result;
    }

    bool NeuralDerivativeTanhAutoTest()
    {
        bool result = true;

        result = result && NeuralActivateDerivativeAutoTest(EPS, true, 3.0f, FUNC_AD(Simd::Base::NeuralDerivativeTanh), FUNC_AD(SimdNeuralDerivativeTanh));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 3.0f, FUNC_AD(Simd::Sse2::NeuralDerivativeTanh), FUNC_AD(SimdNeuralDerivativeTanh));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 3.0f, FUNC_AD(Simd::Avx::NeuralDerivativeTanh), FUNC_AD(SimdNeuralDerivativeTanh));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 3.0f, FUNC_AD(Simd::Avx512f::NeuralDerivativeTanh), FUNC_AD(SimdNeuralDerivativeTanh));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 3.0f, FUNC_AD(Simd::Neon::NeuralDerivativeTanh), FUNC_AD(SimdNeuralDerivativeTanh));
#endif

        return result;
    }

    bool NeuralDerivativeReluAutoTest()
    {
        bool result = true;

        result = result && NeuralActivateDerivativeAutoTest(EPS, true, 0.5f, FUNC_AD(Simd::Base::NeuralDerivativeRelu), FUNC_AD(SimdNeuralDerivativeRelu));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 0.5f, FUNC_AD(Simd::Sse2::NeuralDerivativeRelu), FUNC_AD(SimdNeuralDerivativeRelu));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 0.5f, FUNC_AD(Simd::Avx::NeuralDerivativeRelu), FUNC_AD(SimdNeuralDerivativeRelu));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 0.5f, FUNC_AD(Simd::Avx512f::NeuralDerivativeRelu), FUNC_AD(SimdNeuralDerivativeRelu));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralActivateDerivativeAutoTest(EPS, true, 0.5f, FUNC_AD(Simd::Neon::NeuralDerivativeRelu), FUNC_AD(SimdNeuralDerivativeRelu));
#endif

        return result;
    }

    namespace
    {
        struct FuncUW
        {
            typedef void(*FuncPtr)(const float * x, size_t size, const float * a, const float * b, float * d, float * w);

            FuncPtr func;
            String description;

            FuncUW(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & x, float a, float b, const View & d, const View & w, View & dDst, View & wDst) const
            {
                Simd::Copy(d, dDst);
                Simd::Copy(w, wDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)x.data, x.width, &a, &b, (float*)dDst.data, (float*)wDst.data);
            }
        };
    }
#define FUNC_UW(function) FuncUW(function, #function)

    bool NeuralUpdateWeightsAutoTest(int size, float error, bool relative, const FuncUW & f1, const FuncUW & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View x(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View d(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View w(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(x, -10.0f, 10.0f);
        FillRandom32f(d, -10.0f, 10.0f);
        FillRandom32f(w, -10.0f, 10.0f);

        float a = 2.0, b = 3.0;

        View dDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View wDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View wDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(x, a, b, d, w, dDst1, wDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(x, a, b, d, w, dDst2, wDst2));

        result = result && Compare(dDst1, dDst2, error, true, 32, relative, "d");
        result = result && Compare(wDst1, wDst2, error, true, 32, relative, "w");

        return result;
    }

    bool NeuralUpdateWeightsAutoTest(float error, bool relative, const FuncUW & f1, const FuncUW & f2)
    {
        bool result = true;

        result = result && NeuralUpdateWeightsAutoTest(W*H, error, relative, f1, f2);
        result = result && NeuralUpdateWeightsAutoTest(W*H + O, error, relative, f1, f2);
        result = result && NeuralUpdateWeightsAutoTest(W*H - O, error, relative, f1, f2);

        return result;
    }

    bool NeuralUpdateWeightsAutoTest()
    {
        bool result = true;

        result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Base::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Sse2::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Avx::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Avx512f::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Neon::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));
#endif

        return result;
    }

    namespace
    {
        struct FuncAGU
        {
            typedef void(*FuncPtr)(const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight);

            FuncPtr func;
            String description;

            FuncAGU(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & delta, size_t batch, float alpha, float epsilon, const View & gradientSrc, const View & weightSrc, View & gradientDst, View & weightDst) const
            {
                Simd::Copy(gradientSrc, gradientDst);
                Simd::Copy(weightSrc, weightDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)delta.data, delta.width, batch, &alpha, &epsilon, (float*)gradientDst.data, (float*)weightDst.data);
            }
        };
    }
#define FUNC_AGU(function) FuncAGU(function, #function)

    bool NeuralAdaptiveGradientUpdateAutoTest(int size, float error, bool relative, const FuncAGU & f1, const FuncAGU & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View delta(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View gradientSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View weightSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(delta, -1.0f, 1.0f);
        FillRandom32f(gradientSrc, 0.0f, 0.0001f);
        FillRandom32f(weightSrc, -1.0f, 1.0f);

        const size_t batch = 2;
        const float alpha = 1.0f, epsilon = 0.0001f;

        View gradientDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View weightDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View gradientDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View weightDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(delta, batch, alpha, epsilon, gradientSrc, weightSrc, gradientDst1, weightDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(delta, batch, alpha, epsilon, gradientSrc, weightSrc, gradientDst2, weightDst2));

        result = result && Compare(gradientDst1, gradientDst2, error, true, 32, relative, "gradient");
        result = result && Compare(weightDst1, weightDst2, error, true, 32, relative, "weight");

        return result;
    }

    bool NeuralAdaptiveGradientUpdateAutoTest(float error, bool relative, const FuncAGU & f1, const FuncAGU & f2)
    {
        bool result = true;

        result = result && NeuralAdaptiveGradientUpdateAutoTest(W*H, error, relative, f1, f2);
        result = result && NeuralAdaptiveGradientUpdateAutoTest(W*H + O, error, relative, f1, f2);
        result = result && NeuralAdaptiveGradientUpdateAutoTest(W*H - O, error, relative, f1, f2);

        return result;
    }

    bool NeuralAdaptiveGradientUpdateAutoTest()
    {
        bool result = true;

        result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Base::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Sse2::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Avx::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Avx512f::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Neon::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));
#endif

        return result;
    }

    namespace
    {
        struct FuncC2
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncC2(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const Size & size, const float * weights, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), size.x, size.y, weights, (float*)dstDst.data, dstDst.stride / sizeof(float));
            }
        };
    }
#define FUNC_C2(function) FuncC2(function, #function)

    bool NeuralAddConvolutionAutoTest(const Size & size, float eps, const Size & core, bool forward, const FuncC2 & f1, const FuncC2 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size.x << ", " << size.y << "].");

        Size s(size), d(size);
        if (forward)
            s += core - Size(1, 1);
        else
            d += core - Size(1, 1);

        View src(s.x, s.y, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(src, 0, 1);

        View weights(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(weights, -1, 1);

        View dstSrc(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(dstSrc, -1000, 1000);

        View dstDst1(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));
        View dstDst2(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, size, (float*)weights.data, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, size, (float*)weights.data, dstSrc, dstDst2));

        result = Compare(dstDst1, dstDst2, eps, true, 32, false);

        return result;
    }

    bool NeuralAddConvolutionAutoTest(float eps, const Size & core, bool forward, const FuncC2 & f1, const FuncC2 & f2)
    {
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(Size(W, H), eps, core, forward, f1, f2);
        result = result && NeuralAddConvolutionAutoTest(Size(W - O, H + O), eps, core, forward, f1, f2);
        result = result && NeuralAddConvolutionAutoTest(Size(W + O, H - O), eps, core, forward, f1, f2);

        return result;
    }

    bool NeuralAddConvolution2x2ForwardAutoTest()
    {
        Size core(2, 2);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Base::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Sse2::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx2::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx512f::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Neon::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));
#endif

        return result;
    }

    bool NeuralAddConvolution3x3ForwardAutoTest()
    {
        Size core(3, 3);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Base::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Sse2::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx2::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx512f::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Neon::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));
#endif

        return result;
    }

    bool NeuralAddConvolution4x4ForwardAutoTest()
    {
        Size core(4, 4);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Base::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Sse2::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx2::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx512f::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Neon::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));
#endif

        return result;
    }

    bool NeuralAddConvolution5x5ForwardAutoTest()
    {
        Size core(5, 5);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Base::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Sse2::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx2::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx512f::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Neon::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));
#endif

        return result;
    }

    bool NeuralAddConvolution2x2BackwardAutoTest()
    {
        Size core(2, 2);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Base::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Sse2::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx2::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx512f::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Neon::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));
#endif

        return result;
    }

    bool NeuralAddConvolution3x3BackwardAutoTest()
    {
        Size core(3, 3);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Base::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Sse2::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx2::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx512f::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Neon::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));
#endif

        return result;
    }

    bool NeuralAddConvolution4x4BackwardAutoTest()
    {
        Size core(4, 4);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Base::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Sse2::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx2::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx512f::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Neon::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));
#endif

        return result;
    }

    bool NeuralAddConvolution5x5BackwardAutoTest()
    {
        Size core(5, 5);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Base::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Sse2::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx2::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx512f::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Neon::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));
#endif

        return result;
    }

    namespace
    {
        struct FuncCS
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);

            FuncPtr func;
            String description;

            FuncCS(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const View & dst, const View & sumsSrc, View & sumsDst) const
            {
                Simd::Copy(sumsSrc, sumsDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), (float*)dst.data, dst.stride / sizeof(float), dst.width, dst.height, (float*)sumsDst.data);
            }
        };
    }
#define FUNC_CS(function) FuncCS(function, #function)

    bool NeuralAddConvolutionSumAutoTest(const Size & size, float eps, const Size & core, const FuncCS & f1, const FuncCS & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size.x << ", " << size.y << "].");

        View src(size.x + core.x - 1, size.y + core.y - 1, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(src, -1, 1);

        View dst(size.x, size.y, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(dst, -1, 1);

        View sumsSrc(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(sumsSrc, 3000, 3000);

        View sumsDst1(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        View sumsDst2(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst, sumsSrc, sumsDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst, sumsSrc, sumsDst2));

        result = Compare(sumsDst1, sumsDst2, eps, true, 32);

        return result;
    }

    bool NeuralAddConvolutionSumAutoTest(float eps, const Size & core, const FuncCS & f1, const FuncCS & f2)
    {
        bool result = true;

        result = result && NeuralAddConvolutionSumAutoTest(Size(W, H), eps, core, f1, f2);
        result = result && NeuralAddConvolutionSumAutoTest(Size(W - O, H + O), eps, core, f1, f2);
        result = result && NeuralAddConvolutionSumAutoTest(Size(W + O, H - O), eps, core, f1, f2);

        return result;
    }

    bool NeuralAddConvolution2x2SumAutoTest()
    {
        Size core(2, 2);
        bool result = true;

        result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Base::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse2::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse41::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx2::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx512f::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Neon::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif

        return result;
    }

    bool NeuralAddConvolution3x3SumAutoTest()
    {
        Size core(3, 3);
        bool result = true;

        result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Base::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse2::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse41::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx2::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx512f::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Neon::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif

        return result;
    }

    bool NeuralAddConvolution4x4SumAutoTest()
    {
        Size core(4, 4);
        bool result = true;

        result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Base::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse2::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse41::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx2::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx512f::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Neon::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif

        return result;
    }

    bool NeuralAddConvolution5x5SumAutoTest()
    {
        Size core(5, 5);
        bool result = true;

        result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Base::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse2::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse41::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx2::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx512f::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Neon::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif

        return result;
    }

    namespace
    {
        struct FuncM
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncM(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), src.width, src.height, (float*)dst.data, dst.stride / sizeof(float));
            }
        };
    }
#define FUNC_M(function) FuncM(function, #function)

    bool NeuralPoolingMaxAutoTest(const Size & srcSize, const Size & stride, const Size & pooling, const Size & pad, float eps, const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcSize.x << ", " << srcSize.y << "].");

        View src(srcSize.x, srcSize.y, View::Float, NULL, TEST_ALIGN(srcSize.x));
        FillRandom32f(src, -1, 1);

        Size dstSize((srcSize - pooling + 2 * stride + 2 * pad - Size(1, 1)) / stride);
        View dst1(dstSize.x, dstSize.y, View::Float, NULL, TEST_ALIGN(srcSize.x));
        View dst2(dstSize.x, dstSize.y, View::Float, NULL, TEST_ALIGN(srcSize.x));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = Compare(dst1, dst2, eps, true, 32);

        return result;
    }

    bool NeuralPoolingMaxAutoTest(const Size & stride, const Size & pooling, const Size & pad, float eps, const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        result = result && NeuralPoolingMaxAutoTest(Size(W, H), stride, pooling, pad, eps, f1, f2);
        result = result && NeuralPoolingMaxAutoTest(Size(W + O, H - O), stride, pooling, pad, eps, f1, f2);

        return result;
    }

    bool NeuralPooling1x1Max3x3AutoTest()
    {
        bool result = true;
        Size stride(1, 1), pooling(3, 3), pad(1, 1);

        result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Base::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Sse2::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx2::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx512f::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Neon::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));
#endif

        return result;
    }

    bool NeuralPooling2x2Max2x2AutoTest()
    {
        bool result = true;
        Size stride(2, 2), pooling(2, 2), pad(0, 0);

        result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Base::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::DF)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Sse2::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable && W >= Simd::Avx::DF)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable && W >= Simd::Avx512f::DF)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx512f::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Neon::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));
#endif

        return result;
    }

    bool NeuralPooling2x2Max3x3AutoTest()
    {
        bool result = true;
        Size stride(2, 2), pooling(3, 3), pad(0, 0);

        result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Base::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Sse2::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx2::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx512f::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Neon::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));
#endif

        return result;
    }

    typedef Simd::Neural::Index Index;
    typedef Simd::Neural::Vector Vector;

    namespace
    {
        struct FuncCF
        {
            typedef void(*FuncPtr)(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                const float * weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY,
                void * buffer, size_t * size, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add);

            FuncPtr func;
            String description;

            FuncCF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            static Index DstIndex(const Index & src, size_t dstDepth, const Size & kernel, const Size & pad, const Size & stride, const Size & dilation)
            {
                size_t w = (src.width + 2 * pad.x - (dilation.x * (kernel.x - 1) + 1)) / stride.x + 1;
                size_t h = (src.height + 2 * pad.y - (dilation.y * (kernel.y - 1) + 1)) / stride.y + 1;
                return Index(w, h, dstDepth);
            }

            void Update(const Index & srcIndex, const Size & kernel, const Size & pad, const Size & stride, const Size & dilation, const Index & dstIndex, int add)
            {
                std::stringstream ss;
                ss << description;
                ss << "[" << srcIndex.depth << "x" << srcIndex.height << "x" << srcIndex.width;
#if 0
                ss << "-" << kernel.x << "x" << kernel.y << "-" << pad.x << "x" << pad.y;
                ss << "-" << stride.x << "x" << stride.y << "-" << dilation.x << "x" << dilation.y;
                ss << "-" << dstIndex.width << "x" << dstIndex.height << "x" << dstIndex.depth << "]-" << add;
#else
                ss << "-" << dstIndex.depth << "x" << kernel.y << "x" << kernel.x;
                ss << "-" << stride.x << "-" << pad.x << "-1]";
#endif
                description = ss.str();
            }

            void Call(const Vector & src, const Index & srcIndex, const Vector & weight, const Size & kernel, const Size & pad, const Size & stride, const Size & dilation,
                Vector & buffer, const Vector & dstSrc, Vector & dstDst, const Index & dstIndex, int add) const
            {
                if (add)
                    memcpy(dstDst.data(), dstSrc.data(), dstDst.size() * sizeof(float));
                size_t size = buffer.size() * sizeof(float);
                TEST_PERFORMANCE_TEST(description);
                func(src.data(), srcIndex.width, srcIndex.height, srcIndex.depth,
                    weight.data(), kernel.x, kernel.y, pad.x, pad.y, stride.x, stride.y, dilation.x, dilation.y,
                    buffer.data(), &size, dstDst.data(), dstIndex.width, dstIndex.height, dstIndex.depth, add);
            }
        };
    }
#define FUNC_CF(function) FuncCF(function, #function)

    bool NeuralConvolutionForwardAutoTest(const Index & srcIndex, size_t dstDepth, const Size & kernel, const Size & pad, const Size & stride, const Size & dilation, int add, float eps, FuncCF f1, FuncCF f2)
    {
        bool result = true;

        Index dstIndex = FuncCF::DstIndex(srcIndex, dstDepth, kernel, pad, stride, dilation);

        f1.Update(srcIndex, kernel, pad, stride, dilation, dstIndex, add);
        f2.Update(srcIndex, kernel, pad, stride, dilation, dstIndex, add);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " .");

        Test::PerformanceMeasurerStorage::s_storage.Align(SIMD_ALIGN*(srcIndex.depth%SIMD_ALIGN == 0));

        Vector src(srcIndex.Volume());
        Vector weight(kernel.x*kernel.y*srcIndex.depth*dstIndex.depth);
        Vector dstSrc(dstIndex.Volume());
        Vector dstDst1(dstIndex.Volume());
        Vector dstDst2(dstIndex.Volume());
        Vector buffer(dstIndex.Area()*srcIndex.depth*kernel.x*kernel.y * 2 + dstIndex.Area() * 2);

        FillRandom(src, 0, 1);
        FillRandom(weight, -1, 1);
        const float level = 100;
        FillRandom(dstSrc, level, level);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcIndex, weight, kernel, pad, stride, dilation, buffer, dstSrc, dstDst1, dstIndex, add));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcIndex, weight, kernel, pad, stride, dilation, buffer, dstSrc, dstDst2, dstIndex, add));

        result = Compare(dstDst1, dstDst2, eps, true, 32, false);

        return result;
    }

    bool NeuralConvolutionForwardAutoTest(float eps, const FuncCF & f1, const FuncCF & f2)
    {
        bool result = true;
        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _5(5, 5), _7(7, 7);

#ifdef NDEBUG
#if 1
        //result = result && NeuralConvolutionForwardAutoTest(Index(16, 16, 1), 8, _3, _0, _1, _1, 0, eps, f1, f2);
        //result = result && NeuralConvolutionForwardAutoTest(Index(14, 14, 8), 12, _3, _0, _1, _1, 0, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(6, 6, 12), 24, _3, _0, _1, _1, 0, eps, f1, f2);
        //result = result && NeuralConvolutionForwardAutoTest(Index(4, 4, 24), 32, _1, _0, _1, _1, 0, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(16, 16, 1), 12, _5, _0, _1, _1, 0, eps, f1, f2);
#endif
#if 0
        result = result && NeuralConvolutionForwardAutoTest(Index(320, 180, 3), 10, _3, _0, _1, _1, 0, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(159, 89, 10), 16, _3, _0, _1, _1, 0, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(157, 87, 16), 32, _3, _0, _1, _1, 0, eps, f1, f2);
#endif
#if 0
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 576), 160, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 160), 960, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 960), 160, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 960), 320, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 320), 1280, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 1280), 256, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(5, 5, 256), 512, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(5, 5, 512), 128, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(3, 3, 128), 256, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(3, 3, 256), 128, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(2, 2, 128), 256, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(2, 2, 256), 128, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(2, 2, 256), 64, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(1, 1, 64), 128, _1, _0, _1, _1, 1, eps, f1, f2);
#endif
#if 0
        result = result && NeuralConvolutionForwardAutoTest(Index(256, 256, 48), 48, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(128, 128, 96), 96, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(64, 64, 192), 192, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(32, 32, 384), 384, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(16, 16, 768), 768, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(8, 8, 1536), 1536, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(4, 4, 3072), 3072, _1, _0, _1, _1, 1, eps, f1, f2);

        result = result && NeuralConvolutionForwardAutoTest(Index(256, 256, 16), 16, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(128, 128, 32), 32, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(64, 64, 64), 64, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(32, 32, 128), 128, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(16, 16, 256), 256, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(8, 8, 512), 512, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(4, 4, 1024),1024,  _3, _1, _1, _1, 1, eps, f1, f2);

        result = result && NeuralConvolutionForwardAutoTest(Index(256, 256, 10), 10, _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(128, 128, 20), 20, _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(64, 64, 40), 40, _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(32, 32, 80), 80,  _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(16, 16, 160), 160, _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(8, 8, 320), 320, _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(4, 4, 640), 640, _5, _2, _1, _1, 1, eps, f1, f2);
#endif
#else
        result = result && NeuralConvolutionForwardAutoTest(Index(6, 6, 12), 24, _3, _0, _1, _1, 0, eps, f1, f2);
#endif        

        return result;
    }

    bool NeuralConvolutionForwardAutoTest()
    {
        bool result = true;

        result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Base::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Sse41::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Avx::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Avx2::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Avx512f::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Neon::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool NeuralConvertDataTest(bool create, int width, int height, float eps, const FuncC1 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        View dst1(width*height, 1, View::Float, NULL, TEST_ALIGN(width));
        View dst2(width*height, 1, View::Float, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = Compare(dst1, dst2, eps, true);
        }

        return result;
    }

    bool NeuralConvertDataTest(bool create)
    {
        bool result = true;

        result = result && NeuralConvertDataTest(create, DW, DH, EPS, FUNC_C1(SimdNeuralConvert, true));

        return result;
    }

    bool NeuralProductSumDataTest(bool create, int size, float eps, const FuncPS & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View a(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View b(size, 1, View::Float, NULL, TEST_ALIGN(size));

        float s1, s2;

        if (create)
        {
            FillRandom32f(a);
            FillRandom32f(b);

            TEST_SAVE(a);
            TEST_SAVE(b);

            f.Call(a, b, &s1);

            TEST_SAVE(s1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);

            TEST_LOAD(s1);

            f.Call(a, b, &s2);

            TEST_SAVE(s2);

            result = result && Compare(s1, s2, eps, true);
        }

        return result;
    }

    bool NeuralProductSumDataTest(bool create)
    {
        bool result = true;

        result = result && NeuralProductSumDataTest(create, DH, EPS, FUNC_PS(SimdNeuralProductSum));

        return result;
    }

    bool NeuralAddVectorMultipliedByValueDataTest(bool create, int size, float eps, const FuncAVMV & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        const float value = 0.3f;

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        if (create)
        {
            FillRandom32f(src);
            FillRandom32f(dstSrc);

            TEST_SAVE(src);
            TEST_SAVE(dstSrc);

            f.Call(src, value, dstSrc, dstDst1);

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            f.Call(src, value, dstSrc, dstDst2);

            TEST_SAVE(dstDst2);

            result = result && Compare(dstDst1, dstDst2, eps, true);
        }

        return result;
    }

    bool NeuralAddVectorMultipliedByValueDataTest(bool create)
    {
        bool result = true;

        result = result && NeuralAddVectorMultipliedByValueDataTest(create, DH, EPS, FUNC_AVMV(SimdNeuralAddVectorMultipliedByValue));

        return result;
    }

    bool NeuralAddVectorDataTest(bool create, int size, float eps, const FuncAddVec & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        if (create)
        {
            FillRandom32f(src);
            FillRandom32f(dstSrc);

            TEST_SAVE(src);
            TEST_SAVE(dstSrc);

            f.Call(src, dstSrc, dstDst1);

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            f.Call(src, dstSrc, dstDst2);

            TEST_SAVE(dstDst2);

            result = result && Compare(dstDst1, dstDst2, eps, true);
        }

        return result;
    }

    bool NeuralAddVectorDataTest(bool create)
    {
        bool result = true;

        result = result && NeuralAddVectorDataTest(create, DH, EPS, FUNC_ADDVEC(SimdNeuralAddVector));

        return result;
    }

    bool NeuralAddValueDataTest(bool create, int size, float eps, const FuncAddVal & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        const float value = 3.14f;

        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        if (create)
        {
            FillRandom32f(dstSrc);

            TEST_SAVE(dstSrc);

            f.Call(value, dstSrc, dstDst1);

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            f.Call(value, dstSrc, dstDst2);

            TEST_SAVE(dstDst2);

            result = result && Compare(dstDst1, dstDst2, eps, true);
        }

        return result;
    }

    bool NeuralAddValueDataTest(bool create)
    {
        bool result = true;

        result = result && NeuralAddValueDataTest(create, DH, EPS, FUNC_ADDVAL(SimdNeuralAddValue));

        return result;
    }

    bool NeuralActivateFunctionDataTest(bool create, int size, float error, bool relative, float slope, float lo, float hi, const FuncAF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        if (create)
        {
            FillRandom32f(src, lo, hi);

            TEST_SAVE(src);

            f.Call(src, slope, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, slope, dst2);

            TEST_SAVE(dst2);

            result = Compare(dst1, dst2, error, true, 0, relative);
        }

        return result;
    }

    bool NeuralRoughSigmoidDataTest(bool create)
    {
        return NeuralActivateFunctionDataTest(create, DH, EPS, true, 3.0f, -10.0f, 10.0f, FUNC_AF(SimdNeuralRoughSigmoid));
    }

    bool NeuralRoughSigmoid2DataTest(bool create)
    {
        return NeuralActivateFunctionDataTest(create, DH, EPS, true, 3.0f, -10.0f, 10.0f, FUNC_AF(SimdNeuralRoughSigmoid2));
    }

    bool NeuralRoughTanhDataTest(bool create)
    {
        return NeuralActivateFunctionDataTest(create, DH, EPS, false, 3.0f, -10.0f, 10.0f, FUNC_AF(SimdNeuralRoughTanh));
    }

    bool NeuralPowDataTest(bool create)
    {
        return NeuralActivateFunctionDataTest(create, DH, EPS, false, -0.75f, 0.001f, 9.999f, FUNC_AF(SimdNeuralPow));
    }

    bool NeuralActivateDerivativeDataTest(bool create, int size, float error, bool relative, float slope, const FuncAD & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        if (create)
        {
            FillRandom32f(src, -10.0f, 10.0f);
            FillRandom32f(dstSrc, -1.0f, 1.0f);

            TEST_SAVE(src);
            TEST_SAVE(dstSrc);

            f.Call(src, slope, dstSrc, dstDst1);

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            f.Call(src, slope, dstSrc, dstDst2);

            TEST_SAVE(dstDst2);

            result = Compare(dstDst1, dstDst2, error, true, 0, relative);
        }

        return result;
    }

    bool NeuralDerivativeSigmoidDataTest(bool create)
    {
        bool result = true;

        result = result && NeuralActivateDerivativeDataTest(create, DH, EPS, true, 3.0f, FUNC_AD(SimdNeuralDerivativeSigmoid));

        return result;
    }

    bool NeuralDerivativeTanhDataTest(bool create)
    {
        bool result = true;

        result = result && NeuralActivateDerivativeDataTest(create, DH, EPS, true, 3.0f, FUNC_AD(SimdNeuralDerivativeTanh));

        return result;
    }

    bool NeuralDerivativeReluDataTest(bool create)
    {
        bool result = true;

        result = result && NeuralActivateDerivativeDataTest(create, DH, EPS, true, 0.5f, FUNC_AD(SimdNeuralDerivativeRelu));

        return result;
    }

    bool NeuralUpdateWeightsDataTest(bool create, int size, float error, bool relative, const FuncUW & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View x(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View d(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View w(size, 1, View::Float, NULL, TEST_ALIGN(size));

        float a = 3.0, b = 5.0;

        View dDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View wDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View wDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        if (create)
        {
            FillRandom32f(x, -10.0f, 10.0f);
            FillRandom32f(d, -10.0f, 10.0f);
            FillRandom32f(w, -10.0f, 10.0f);

            TEST_SAVE(x);
            TEST_SAVE(d);
            TEST_SAVE(w);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(x, a, b, d, w, dDst1, wDst1));

            TEST_SAVE(dDst1);
            TEST_SAVE(wDst1);
        }
        else
        {
            TEST_LOAD(x);
            TEST_LOAD(d);
            TEST_LOAD(w);

            TEST_LOAD(dDst1);
            TEST_LOAD(wDst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(x, a, b, d, w, dDst2, wDst2));

            TEST_SAVE(dDst2);
            TEST_SAVE(wDst2);

            result = Compare(dDst1, dDst2, error, true, 32, relative);
            result = Compare(wDst1, wDst2, error, true, 32, relative);
        }

        return result;
    }

    bool NeuralUpdateWeightsDataTest(bool create)
    {
        bool result = true;

        result = result && NeuralUpdateWeightsDataTest(create, DH, EPS, true, FUNC_UW(SimdNeuralUpdateWeights));

        return result;
    }

    bool NeuralAdaptiveGradientUpdateDataTest(bool create, int size, float error, bool relative, const FuncAGU & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View delta(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View gradientSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View weightSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));

        const size_t batch = 64;
        const float alpha = 0.01f, epsilon = 0.0001f;

        View gradientDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View weightDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View gradientDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View weightDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        if (create)
        {
            FillRandom32f(delta, -1.0f, 1.0f);
            FillRandom32f(gradientSrc, 0.0f, 1.0f);
            FillRandom32f(weightSrc, -1.0f, 1.0f);

            TEST_SAVE(delta);
            TEST_SAVE(gradientSrc);
            TEST_SAVE(weightSrc);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(delta, batch, alpha, epsilon, gradientSrc, weightSrc, gradientDst1, weightDst1));

            TEST_SAVE(gradientDst1);
            TEST_SAVE(weightDst1);
        }
        else
        {
            TEST_LOAD(delta);
            TEST_LOAD(gradientSrc);
            TEST_LOAD(weightSrc);

            TEST_LOAD(gradientDst1);
            TEST_LOAD(weightDst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(delta, batch, alpha, epsilon, gradientSrc, weightSrc, gradientDst2, weightDst2));

            TEST_SAVE(gradientDst2);
            TEST_SAVE(weightDst2);

            result = Compare(gradientDst1, gradientDst2, error, true, 32, relative);
            result = Compare(weightDst1, weightDst2, error, true, 32, relative);
        }

        return result;
    }

    bool NeuralAdaptiveGradientUpdateDataTest(bool create)
    {
        bool result = true;

        result = result && NeuralAdaptiveGradientUpdateDataTest(create, DH, EPS, true, FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));

        return result;
    }

    bool NeuralAddConvolutionDataTest(bool create, const Size & size, float eps, const Size & core, bool forward, const FuncC2 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size.x << ", " << size.y << "].");

        Size s(size), d(size);
        if (forward)
            s += core - Size(1, 1);
        else
            d += core - Size(1, 1);

        View src(s.x, s.y, View::Float, NULL, TEST_ALIGN(size.x));
        View weights(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        View dstSrc(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));
        View dstDst1(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));
        View dstDst2(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));

        if (create)
        {
            FillRandom32f(src, 0, 1);
            FillRandom32f(weights, -1, 1);
            FillRandom32f(dstSrc, 1000, 2000);

            TEST_SAVE(src);
            TEST_SAVE(weights);
            TEST_SAVE(dstSrc);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, size, (float*)weights.data, dstSrc, dstDst1));

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(weights);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, size, (float*)weights.data, dstSrc, dstDst2));

            TEST_SAVE(dstDst2);

            result = Compare(dstDst1, dstDst2, eps, true, 32, true);
        }

        return result;
    }

    bool NeuralAddConvolution2x2ForwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(2, 2), true, FUNC_C2(SimdNeuralAddConvolution2x2Forward));
    }

    bool NeuralAddConvolution3x3ForwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(3, 3), true, FUNC_C2(SimdNeuralAddConvolution3x3Forward));
    }

    bool NeuralAddConvolution4x4ForwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(4, 4), true, FUNC_C2(SimdNeuralAddConvolution4x4Forward));
    }

    bool NeuralAddConvolution5x5ForwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(5, 5), true, FUNC_C2(SimdNeuralAddConvolution5x5Forward));
    }

    bool NeuralAddConvolution2x2BackwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(2, 2), false, FUNC_C2(SimdNeuralAddConvolution2x2Backward));
    }

    bool NeuralAddConvolution3x3BackwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(3, 3), false, FUNC_C2(SimdNeuralAddConvolution3x3Backward));
    }

    bool NeuralAddConvolution4x4BackwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(4, 4), false, FUNC_C2(SimdNeuralAddConvolution4x4Backward));
    }

    bool NeuralAddConvolution5x5BackwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(5, 5), false, FUNC_C2(SimdNeuralAddConvolution5x5Backward));
    }

    bool NeuralAddConvolutionSumDataTest(bool create, const Size & size, float eps, const Size & core, const FuncCS & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size.x << ", " << size.y << "].");

        View src(size.x + core.x - 1, size.y + core.y - 1, View::Float, NULL, TEST_ALIGN(size.x));
        View dst(size.x, size.y, View::Float, NULL, TEST_ALIGN(size.x));
        View sumsSrc(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        View sumsDst1(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        View sumsDst2(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));

        if (create)
        {
            FillRandom32f(src, -1, 1);
            FillRandom32f(dst, -1, 1);
            FillRandom32f(sumsSrc, 2000, 3000);

            TEST_SAVE(src);
            TEST_SAVE(dst);
            TEST_SAVE(sumsSrc);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, dst, sumsSrc, sumsDst1));

            TEST_SAVE(sumsDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(dst);
            TEST_LOAD(sumsSrc);

            TEST_LOAD(sumsDst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, dst, sumsSrc, sumsDst2));

            TEST_SAVE(sumsDst2);

            result = Compare(sumsDst1, sumsDst2, eps, true, 32);
        }

        return result;
    }

    bool NeuralAddConvolution2x2SumDataTest(bool create)
    {
        return NeuralAddConvolutionSumDataTest(create, Size(DW, DH), EPS, Size(2, 2), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
    }

    bool NeuralAddConvolution3x3SumDataTest(bool create)
    {
        return NeuralAddConvolutionSumDataTest(create, Size(DW, DH), EPS, Size(3, 3), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
    }

    bool NeuralAddConvolution4x4SumDataTest(bool create)
    {
        return NeuralAddConvolutionSumDataTest(create, Size(DW, DH), EPS, Size(4, 4), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
    }

    bool NeuralAddConvolution5x5SumDataTest(bool create)
    {
        return NeuralAddConvolutionSumDataTest(create, Size(DW, DH), EPS, Size(5, 5), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
    }

    bool NeuralPoolingMaxDataTest(bool create, const Size & srcSize, const Size & stride, const Size & pooling, const Size & pad, float eps, const FuncM & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << srcSize.x << ", " << srcSize.y << "].");

        Size dstSize((srcSize - pooling + 2 * stride + 2 * pad - Size(1, 1)) / stride);
        View src(srcSize.x, srcSize.y, View::Float, NULL, TEST_ALIGN(srcSize.x));
        View dst1(dstSize.x, dstSize.y, View::Float, NULL, TEST_ALIGN(srcSize.x));
        View dst2(dstSize.x, dstSize.y, View::Float, NULL, TEST_ALIGN(srcSize.x));

        if (create)
        {
            FillRandom32f(src, -1, 1);

            TEST_SAVE(src);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, dst1));

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, dst2));

            TEST_SAVE(dst2);

            result = Compare(dst1, dst2, eps, true, 32);
        }

        return result;
    }

    bool NeuralPooling1x1Max3x3DataTest(bool create)
    {
        return NeuralPoolingMaxDataTest(create, Size(DW, DH), Size(1, 1), Size(3, 3), Size(1, 1), EPS, FUNC_M(SimdNeuralPooling1x1Max3x3));
    }

    bool NeuralPooling2x2Max2x2DataTest(bool create)
    {
        return NeuralPoolingMaxDataTest(create, Size(DW, DH), Size(2, 2), Size(2, 2), Size(0, 0), EPS, FUNC_M(SimdNeuralPooling2x2Max2x2));
    }

    bool NeuralPooling2x2Max3x3DataTest(bool create)
    {
        return NeuralPoolingMaxDataTest(create, Size(DW, DH), Size(2, 2), Size(3, 3), Size(0, 0), EPS, FUNC_M(SimdNeuralPooling2x2Max3x3));
    }

    bool NeuralConvolutionForwardDataTest(bool create, const Index & srcIndex, const Size & kernel, const Size & pad, const Size & stride, const Size & dilation, int add, float eps, FuncCF f)
    {
        bool result = true;

        Index dstIndex = FuncCF::DstIndex(srcIndex, srcIndex.depth, kernel, pad, stride, dilation);

        f.Update(srcIndex, kernel, pad, stride, dilation, dstIndex, add);

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " .");

        Test::PerformanceMeasurerStorage::s_storage.Align(SIMD_ALIGN);

        Vector src(srcIndex.Volume());
        Vector weight(kernel.x*kernel.y*srcIndex.depth*dstIndex.depth);
        Vector dstSrc(dstIndex.Volume());
        Vector dstDst1(dstIndex.Volume());
        Vector dstDst2(dstIndex.Volume());
        Vector buffer;

        if (create)
        {
            FillRandom(src, 0, 1);
            FillRandom(weight, -1, 1);
            FillRandom(dstSrc, -1000, 1000);

            TEST_SAVE(src);
            TEST_SAVE(weight);
            TEST_SAVE(dstSrc);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, srcIndex, weight, kernel, pad, stride, dilation, buffer, dstSrc, dstDst1, dstIndex, add));

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(weight);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, srcIndex, weight, kernel, pad, stride, dilation, buffer, dstSrc, dstDst2, dstIndex, add));

            TEST_SAVE(dstDst2);

            result = Compare(dstDst1, dstDst2, eps, true, 32);
        }

        return result;
    }

    bool NeuralConvolutionForwardDataTest(bool create)
    {
        Size _1(1, 1), _3(3, 3);
        return NeuralConvolutionForwardDataTest(create, Index(64, 64, 4), _3, _1, _1, _1, 1, EPS, FUNC_CF(SimdNeuralConvolutionForward));
    }
}

//-----------------------------------------------------------------------------

namespace Test
{
    typedef Simd::Neural::Vector Vector;
    typedef Simd::Neural::Vectors Vectors;
    typedef Simd::Neural::Label Label;
    typedef Simd::Neural::Labels Labels;
    typedef Simd::Neural::VectorI VectorI;
    typedef Simd::Neural::Network Network;
    typedef std::pair<float, float> Error;

    struct TrainSample
    {
        Vectors src;
        Labels lbl;
        Vectors dst;

        void Resize(size_t size)
        {
            src.resize(size);
            lbl.resize(size);
            dst.resize(size);
        }

        void Reserve(size_t size)
        {
            src.reserve(size);
            lbl.reserve(size);
            dst.reserve(size);
        }
    };

    struct TrainData
    {
        TrainSample train;
        TrainSample check;
    };

    struct TrainOptions : public Simd::Neural::TrainOptions
    {
        float threshold;
        size_t logEvery;

        TrainOptions()
            : Simd::Neural::TrainOptions()
            , threshold(0.5f)
            , logEvery(10)
        {
        }
    };

    Error Check(Network & net, const TrainSample & sample, float positive, bool train)
    {
        double sum = 0;
        size_t count = 0, size = net.OutputIndex().Volume();
        for (size_t i = 0; i < sample.src.size(); ++i)
        {
            const Vector & dst = sample.dst[i];
            Label lbl = sample.lbl[i];

            Vector cur = net.Predict(sample.src[i], 0, train ? Simd::Neural::Layer::Check : Simd::Neural::Layer::Fast);

            float difference = 0;
            for (size_t j = 0; j < size; ++j)
                difference += Simd::Square(dst[j] - cur[j]);
            sum += difference / size;

            float negative = lbl < size ? dst[lbl] : positive;
            for (size_t j = 0; j < size; ++j)
            {
                if (j == lbl)
                {
                    if (cur[j] < positive)
                    {
                        count++;
                        break;
                    }
                }
                else
                {
                    if (cur[j] > negative)
                    {
                        count++;
                        break;
                    }
                }
            }
        }
        return Error((float)::sqrt(sum / sample.src.size()), float(count) / float(sample.src.size()));
    }

    struct Logger
    {
        void operator() ()
        {
            if (_network && _data && _options)
            {
                if (_current%_options->logEvery == 0)
                {
                    Error train = Check(*_network, _data->train, _options->threshold, true);
                    Error check = Check(*_network, _data->check, _options->threshold, true);
                    TEST_LOG_SS(Info, std::setprecision(6) << std::fixed << "Epoch " << _current
                        << ": train (value = " << train.first << " ; count = " << train.second << ")"
                        << ", check (value = " << check.first << " ; count = " << check.second << ").");
                }
                else
                    std::cout << "Epoch " << _current << "\r";

                _current++;
            }
        }

        Logger(Network * network = NULL, TrainData * data = NULL, TrainOptions * options = NULL)
            : _current(options ? options->epochStart : 0)
            , _network(network)
            , _options(options)
            , _data(data)
        {
        }

    private:
        size_t _current;
        Network * _network;
        TrainOptions * _options;
        TrainData *_data;
    };

    bool LoadDigits(const Network & net, bool error, TrainSample & dst)
    {
        Size size = net.InputIndex().Size();
        dst.Resize(0);
        for (size_t i = 0, n = 10, current = 0, total = 0; (error ? i <= n : i < n); ++i)
        {
            String path = (i < n ? ROOT_PATH + "/data/image/digit/" + char('0' + i) + ".pgm" : ROOT_PATH + "/data/image/face/lena.pgm");
            View pooled;
            if (!pooled.Load(path))
            {
                TEST_LOG_SS(Error, "Can't load test image '" << path << "' !");
                return false;
            }
            Size number = pooled.Size() / size, shift;
            total += number.x*number.y;
            dst.Resize(total);
            for (shift.y = 0; shift.y < number.y; ++shift.y)
            {
                for (shift.x = 0; shift.x < number.x; ++shift.x, ++current)
                {
                    dst.lbl[current] = i;
                    dst.src[current].resize(size.x*size.y);
                    Simd::NeuralConvert(pooled.Region(shift*size, shift*size + size), dst.src[current].data(), size.x, true);
                }
            }
        }
        net.Convert(dst.lbl, dst.dst);
        return true;
    }

#define TEST_ADD_LAYER(net, layer) \
    if(!net.Add(layer)) \
    { \
       std::cout << "Can't add layer '" << #layer "' to network!" << std::endl; \
       return false; \
    }

#define SIMD_NEURAL_EXPERIMENT_VERSION 4

    bool CreateNetwork(Network & net, bool dropout, bool experimental)
    {
        using namespace Simd::Neural;
        net.Clear();
        if (experimental)
        {
#if SIMD_NEURAL_EXPERIMENT_VERSION == 0 // not square shape of convolutional core.
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(16, 16), 1, 12, Size(5, 3))));
            TEST_ADD_LAYER(net, (new MaxPoolingLayer(Function::Relu, Size(12, 14), 12, Size(2, 2), Size(2, 2))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(6, 7), 12, 24, Size(3, 3))));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Relu, 4 * 5 * 24, 96)));
            if (dropout)
                TEST_ADD_LAYER(net, (new DropoutLayer(96, 0.9f)));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Sigmoid, 96, 10)));
#elif SIMD_NEURAL_EXPERIMENT_VERSION == 1 // using of average layer.
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(16, 16), 1, 12, Size(5, 5))));
            TEST_ADD_LAYER(net, (new AveragePoolingLayer(Function::Relu, Size(12, 12), 12, Size(2, 2), Size(2, 2))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(6, 6), 12, 24, Size(3, 3))));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Relu, 4 * 4 * 24, 96)));
            if (dropout)
                TEST_ADD_LAYER(net, (new DropoutLayer(96, 0.9f)));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Sigmoid, 96, 10)));
#elif SIMD_NEURAL_EXPERIMENT_VERSION == 2 // using of convolutional layer with core 1x1.
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(16, 16), 1, 12, Size(5, 5))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(12, 12), 12, 12, Size(1, 1))));
            TEST_ADD_LAYER(net, (new MaxPoolingLayer(Function::Relu, Size(12, 12), 12, Size(2, 2), Size(2, 2))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(6, 6), 12, 24, Size(3, 3))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(4, 4), 24, 24, Size(1, 1))));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Relu, 4 * 4 * 24, 96)));
            if (dropout)
                TEST_ADD_LAYER(net, (new DropoutLayer(96, 0.9f)));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Sigmoid, 96, 10)));
#elif SIMD_NEURAL_EXPERIMENT_VERSION == 3 // using of convolutional layer with core 2x2 and 4x4.
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(16, 16), 1, 8, Size(4, 4))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(13, 13), 8, 12, Size(2, 2))));
            TEST_ADD_LAYER(net, (new MaxPoolingLayer(Function::Relu, Size(12, 12), 12, Size(2, 2), Size(2, 2))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(6, 6), 12, 18, Size(2, 2))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(5, 5), 18, 24, Size(2, 2))));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Relu, 4 * 4 * 24, 96)));
            if (dropout)
                TEST_ADD_LAYER(net, (new DropoutLayer(96, 0.9f)));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Sigmoid, 96, 10)));
#elif SIMD_NEURAL_EXPERIMENT_VERSION == 4 // using of max pooling layer with size 3x3.
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(16, 16), 1, 12, Size(4, 4))));
            TEST_ADD_LAYER(net, (new MaxPoolingLayer(Function::Relu, Size(13, 13), 12, Size(3, 3), Size(2, 2))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(6, 6), 12, 18, Size(2, 2))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(5, 5), 18, 24, Size(2, 2))));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Relu, 4 * 4 * 24, 96)));
            if (dropout)
                TEST_ADD_LAYER(net, (new DropoutLayer(96, 0.9f)));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Sigmoid, 96, 10)));
#endif
        }
        else
        {
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(16, 16), 1, 12, Size(5, 5))));
            TEST_ADD_LAYER(net, (new MaxPoolingLayer(Function::Relu, Size(12, 12), 12, Size(2, 2), Size(2, 2))));
            TEST_ADD_LAYER(net, (new ConvolutionalLayer(Function::Relu, Size(6, 6), 12, 24, Size(3, 3))));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Relu, 4 * 4 * 24, 96)));
            if (dropout)
                TEST_ADD_LAYER(net, (new DropoutLayer(96, 0.9f)));
            TEST_ADD_LAYER(net, (new FullyConnectedLayer(Function::Sigmoid, 96, 10)));
        }
        return true;
    }

    bool NeuralPredictSpecialTest()
    {
        Network net;
        if (!CreateNetwork(net, false, false))
        {
            TEST_LOG_SS(Error, "Can't create Simd::Neural::Network!");
            return false;
        }

        String path = ROOT_PATH + "/data/network/digit.txt";
        if (!net.Load(path))
        {
            TEST_LOG_SS(Error, "Can't load Simd::Neural::Network from file '" << path << "'!");
            return false;
        }

        TrainSample sample;
        if (!LoadDigits(net, true, sample))
            return false;

        Error error;
        for(size_t i = 0; i < 100; i++)
            error = Check(net, sample, 0.5, false);
        TEST_LOG_SS(Info, std::setprecision(6) << "Predict error : (value = " << error.first << " ; count = " << error.second << ")." << std::endl);

#ifdef TEST_PERFORMANCE_TEST_ENABLE
        TEST_LOG_SS(Info, PerformanceMeasurerStorage::s_storage.ConsoleReport(false, true));
        PerformanceMeasurerStorage::s_storage.Clear();
#endif

        return true;
    }

    SIMD_INLINE void Add(const TrainSample & src, size_t index, TrainSample & dst)
    {
        dst.src.push_back(src.src[index]);
        dst.lbl.push_back(src.lbl[index]);
        dst.dst.push_back(src.dst[index]);
    }

    void Prepare(const TrainSample & src, size_t checkEvery, TrainData & dst)
    {
        VectorI index(src.src.size());
        for (size_t i = 0; i < index.size(); ++i)
            index[i] = i;
#ifdef SIMD_CPP_2017_ENABLE
        std::random_device device;
        std::minstd_rand generator(device());
        std::shuffle(index.begin(), index.end(), generator);
#else
        std::random_shuffle(index.begin(), index.end());
#endif
        dst.check.Reserve(index.size());
        dst.train.Reserve(index.size());
        for (size_t i = 0; i < index.size(); ++i)
        {
            size_t idx = index[i];
            if (i%checkEvery == 0)
                Add(src, idx, dst.check);
            else
                Add(src, idx, dst.train);
        }
        TEST_LOG_SS(Info, "Simd::Neural::Network uses " << dst.train.src.size() << " samples for train and " << dst.check.src.size() << " samples for check.");
    }

    bool NeuralTrainSpecialTest()
    {
        Network net;
        if (!CreateNetwork(net, false, false))
        {
            TEST_LOG_SS(Error, "Can't create Simd::Neural::Network!");
            return false;
        }

        TrainSample sample;
        if (!LoadDigits(net, true, sample))
            return false;

        TrainData data;
        Prepare(sample, 8, data);

        TrainOptions options;
        options.epochFinish = 101;
#ifdef _DEBUG
        options.threadNumber = 1;
#endif

        Logger logger(&net, &data, &options);

        net.Train(data.train.src, data.train.dst, options, logger);

#ifdef TEST_PERFORMANCE_TEST_ENABLE
        TEST_LOG_SS(Info, PerformanceMeasurerStorage::s_storage.ConsoleReport(false, true));
        PerformanceMeasurerStorage::s_storage.Clear();
#endif

        return true;
    }
}

