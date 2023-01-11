#include <math.h>
#include <parsim.h>

#include <algorithm>

#include "Math.h"

// Cumulative normal distribution function
STATIC_INLINE float CND(float X) {
    float L = fabsf(X);

    float k = 1.f / (1.f + 0.2316419f * L);
    float k2 = k * k;
    float k3 = k2 * k;
    float k4 = k2 * k2;
    float k5 = k3 * k2;

    const float invSqrt2Pi = 0.39894228040f;
    float w = (0.31938153f * k - 0.356563782f * k2 + 1.781477937f * k3 +
               -1.821255978f * k4 + 1.330274429f * k5);
    w *= invSqrt2Pi * expf(-L * L * .5f);

    if (X > 0.f) w = 1.f - w;
    return w;
}

void black_scholes_psv(float Sa[], float Xa[], float Ta[], float ra[],
                       float va[], float result[], int count) {
#psim num_spmd_threads(count) gang_size(32)
    {
        uint64_t i = psim_get_thread_num();
        float S = Sa[i], X = Xa[i];
        float T = Ta[i], r = ra[i];
        float v = va[i];

        float d1 = (logf(S / X) + (r + v * v * .5f) * T) / (v * sqrtf(T));
        float d2 = d1 - v * sqrtf(T);

        result[i] = S * CND(d1) - X * expf(-r * T) * CND(d2);
    }
}

STATIC_INLINE float binomial_put(float S, float X, float T, float r, float v) { 
    float V[BINOMIAL_NUM];
    float dt = T / BINOMIAL_NUM;
    float u = expf(v * sqrt(dt));
    float d = 1. / u;
    float disc = expf(r * dt);
    float Pu = (disc - d) / (u - d);

    for (int j = 0; j < BINOMIAL_NUM; ++j) {
        float upow = powf(u, (float)(2 * j - BINOMIAL_NUM));
        V[j] = fmaxf(0.f, X - S * upow);
    }

    for (int j = BINOMIAL_NUM - 1; j >= 0; --j)
        for (int k = 0; k < j; ++k)
            V[k] = ((1 - Pu) * V[k] + Pu * V[k + 1]) / disc;
    return V[0];
}

void binomial_put_psv(float Sa[], float Xa[], float Ta[], float ra[],
                      float va[], float result[], int count) {
#psim num_spmd_threads(count) gang_size(16)
    {
        uint64_t i = psim_get_thread_num();
        float S = Sa[i], X = Xa[i], T = Ta[i], r = ra[i], v = va[i];
        result[i] = binomial_put(S, X, T, r, v);
    }
}
