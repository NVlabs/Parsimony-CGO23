/*
  Copyright (c) 2010-2014, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
#endif

#include "../../common/timing.h"
#include "noise_ispc.h"
#include <algorithm>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
using namespace ispc;

extern void noise_serial(float x0, float y0, float x1, float y1, int width, int height, float output[]);
extern void noise_psv(float x0, float y0, float x1, float y1, int width, int height, float output[]);

/* Write a PPM image file with the image */
static void writePPM(float *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width * height; ++i) {
        float v = buf[i] * 255.f;
        if (v < 0)
            v = 0;
        if (v > 255)
            v = 255;
        for (int j = 0; j < 3; ++j)
            fputc((char)v, fp);
    }
    fclose(fp);
}

int main(int argc, char *argv[]) {
    static unsigned int test_iterations[] = {5, 5};
    unsigned int width = 768;
    unsigned int height = 768;
    float x0 = -10;
    float x1 = 10;
    float y0 = -10;
    float y1 = 10;

    if (argc > 1) {
        if (strncmp(argv[1], "--scale=", 8) == 0) {
            float scale = atof(argv[1] + 8);
            width *= scale;
            height *= scale;
        }
    }
    if ((argc == 3) || (argc == 4)) {
        for (int i = 0; i < 2; i++) {
            test_iterations[i] = atoi(argv[argc - 2 + i]);
        }
    }
    float *buf = new float[width * height];

    //
    // Compute the image using the ispc implementation; report the average
    // time across test_iterations.
    //
   double avgISPC = 0;
   for (unsigned int i = 0; i < test_iterations[0]; ++i) {
       reset_and_start_timer();
       noise_ispc(x0, y0, x1, y1, width, height, buf);
        double dt = get_elapsed_mcycles();
       // printf("@time of ISPC run:\t\t\t[%.3f] million cycles\n", dt);
       avgISPC += dt;
    }

   
    writePPM(buf, width, height, "noise-ispc.ppm");

    // Clear out the buffer
    for (unsigned int i = 0; i < width * height; ++i)
       buf[i] = 0;


    //
    // Compute the image using the PSV implementation; report the average
    // time across test_iterations.
    //
    double avgPSV = 0;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
       reset_and_start_timer();
        noise_psv(x0, y0, x1, y1, width, height, buf);
       double dt = get_elapsed_mcycles();
        // printf("@time of PSV run:\t\t\t[%.3f] million cycles\n", dt);
        avgPSV +=  dt;
    }
    
    writePPM(buf, width, height, "noise-psv.ppm");

    // Clear out the buffer
    for (unsigned int i = 0; i < width * height; ++i)
       buf[i] = 0;

    //
    // And run the serial implementation test_iterations times, again reporting the
    // average time.
    //
    double avgSerial = 0;
   for (unsigned int i = 0; i < test_iterations[1]; ++i) {
       reset_and_start_timer();
        noise_serial(x0, y0, x1, y1, width, height, buf);
       double dt = get_elapsed_mcycles();
       // printf("@time of serial run:\t\t\t[%.3f] million cycles\n", dt);
       avgSerial += dt;
    }

    writePPM(buf, width, height, "noise-serial.ppm");

    printf("Perlin Noise Function, %.3f, %.3f, %.3f\n", avgSerial/test_iterations[1], avgISPC/test_iterations[0], avgPSV/test_iterations[0]);

/*    printf("[noise ispc]:\t\t\t[%.3f] million cycles\n", avgISPC/test_iterations[0]);
    printf("[noise [psv]]:\t\t\t[%.3f] million cycles\n", avgPSV/test_iterations[0]);
    printf("[noise serial]:\t\t\t[%.3f] million cycles\n", avgSerial/test_iterations[1]);
    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", avgSerial / avgISPC);
    printf("\t\t\t\t(%.2fx speedup from PSV)\n", avgSerial / avgPSV);*/

    return 0;
}
