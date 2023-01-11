#include <parsim.h>

#include "Math.h"

STATIC_INLINE int mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f) break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

void mandelbrot_psv(float x0, float y0, float x1, float y1, int width,
                    int height, int maxIterations, int output[]) {
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    for (int j = 0; j < height; j++) {
#psim num_spmd_threads(width) gang_size(16)
        {
            uint64_t i = psim_get_thread_num();
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            uint64_t index = (j * width + i);
            output[index] = mandel(x, y, maxIterations);
        }
    }
}
