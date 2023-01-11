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
#include "stencil_ispc.h"
#include <algorithm>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <string.h>
using namespace ispc;

extern void loop_stencil_serial(int t0, int t1, int x0, int x1, int y0, int y1, int z0, int z1, int Nx, int Ny, int Nz,
                                const float coef[5], const float vsq[], float Aeven[], float Aodd[]);

extern void loop_stencil_psv(int t0, int t1, int x0, int x1, int y0, int y1, int z0, int z1, int Nx, int Ny, int Nz,
                                const float coef[5], const float vsq[], float Aeven[], float Aodd[]);

void InitData(int Nx, int Ny, int Nz, float *A[2], float *vsq) {
    int offset = 0;
    for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x, ++offset) {
                A[0][offset] = (x < Nx / 2) ? x / float(Nx) : y / float(Ny);
                A[1][offset] = 0;
                vsq[offset] = x * y * z / float(Nx * Ny * Nz);
            }
}

int main(int argc, char *argv[]) {
    static unsigned int test_iterations[] = {5, 5, 5};
    int Nx = 256, Ny = 256, Nz = 256;
    int width = 4;

    if (argc > 1) {
        if (strncmp(argv[1], "--scale=", 8) == 0) {
            float scale = atof(argv[1] + 8);
            Nx *= scale;
            Ny *= scale;
            Nz *= scale;
        }
    }
    if ((argc == 4) || (argc == 5)) {
        for (int i = 0; i < 3; i++) {
            test_iterations[i] = atoi(argv[argc - 3 + i]);
        }
    }

    float *Aserial[2], *Aispc[2];
    Aserial[0] = new float[Nx * Ny * Nz];
    Aserial[1] = new float[Nx * Ny * Nz];
    Aispc[0] = new float[Nx * Ny * Nz];
    Aispc[1] = new float[Nx * Ny * Nz];
    float *vsq = new float[Nx * Ny * Nz];

    float coeff[4] = {0.5, -.25, .125, -.0625};

    InitData(Nx, Ny, Nz, Aispc, vsq);
    //
    // Compute the image using the ispc implementation on one core; report
    // the average time of test_iterations runs.
    //
    double avgTimeISPC = 0;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        reset_and_start_timer();
        loop_stencil_ispc(0, 6, width, Nx - width, width, Ny - width, width, Nz - width, Nx, Ny, Nz, coeff, vsq,
                          Aispc[0], Aispc[1]);
        double dt = get_elapsed_mcycles();
        // printf("@time of ISPC run:\t\t\t[%.3f] million cycles\n", dt);
        avgTimeISPC += dt;
    }



    InitData(Nx, Ny, Nz, Aispc, vsq);

    //
    // Compute the image using the psv implementation on one core; report
    // the the average time of test_iterations runs.
    //
    double avgTimePSV = 0;
    for (unsigned int i = 0; i < test_iterations[0]; ++i) {
        reset_and_start_timer();
        loop_stencil_psv(0, 6, width, Nx - width, width, Ny - width, width, Nz - width, Nx, Ny, Nz, coeff, vsq,
                          Aispc[0], Aispc[1]);
        double dt = get_elapsed_mcycles();
        // printf("@time of psv run:\t\t\t[%.3f] million cycles\n", dt);
        avgTimePSV += dt;
    }


    InitData(Nx, Ny, Nz, Aispc, vsq);

/*    
    Compute the image using the ispc implementation with tasks; report
    the average time of test_iterations runs.
    
    double minTimeISPCTasks = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; ++i) {
        reset_and_start_timer();
        loop_stencil_ispc_tasks(0, 6, width, Nx - width, width, Ny - width, width, Nz - width, Nx, Ny, Nz, coeff, vsq,
                                Aispc[0], Aispc[1]);
        double dt = get_elapsed_mcycles();
        printf("@time of ISPC + TASKS run:\t\t\t[%.3f] million cycles\n", dt);
        minTimeISPCTasks += dt;
    }
    printf("[stencil ispc + tasks]:\t\t[%.3f] million cycles\n", minTimeISPCTasks);
    InitData(Nx, Ny, Nz, Aserial, vsq);
*/

    //
    // And run the serial implementation test_iterations times, again reporting the
    // the average time.
    //
    double avgTimeSerial = 0;
    for (unsigned int i = 0; i < test_iterations[2]; ++i) {
        reset_and_start_timer();
        loop_stencil_serial(0, 6, width, Nx - width, width, Ny - width, width, Nz - width, Nx, Ny, Nz, coeff, vsq,
                            Aserial[0], Aserial[1]);
        double dt = get_elapsed_mcycles();
        // printf("@time of serial run:\t\t\t[%.3f] million cycles\n", dt);
        avgTimeSerial += dt;
    }

    printf("3D Stencil, %.3f, %.3f, %.3f\n", avgTimeSerial/test_iterations[2], avgTimeISPC/test_iterations[0], avgTimePSV/test_iterations[0]);

/*    printf("[stencil ispc 1 core]:\t\t[%.3f] million cycles\n", avgTimeISPC/test_iterations[0]);
    printf("[stencil psv 1 core]:\t\t[%.3f] million cycles\n", avgTimePSV/test_iterations[0]);
    printf("[stencil serial]:\t\t[%.3f] million cycles\n", avgTimeSerial/test_iterations[2]);
    printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from PSV)\n", avgTimeSerial / avgTimeISPC,
           avgTimeSerial / avgTimePSV);
*/

    // Check for agreement
/*    int offset = 0;
    for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x, ++offset) {
                float error = fabsf((Aserial[1][offset] - Aispc[1][offset]) / Aserial[1][offset]);
                if (error > 1e-4)
                    printf("Error @ (%d,%d,%d): ispc = %f, serial = %f\n", x, y, z, Aispc[1][offset],
                           Aserial[1][offset]);
            }*/

    return 0;
}
