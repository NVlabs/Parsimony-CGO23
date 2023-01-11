/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#define GANG_SIZE 8

#include <math.h>
#include <parsim.h>
#include <stdint.h>
#include <stdio.h>
#include <cassert>
#include <iostream>

#define DEBUG(x) \
    if (0) {     \
        x;       \
    }

static __attribute__((always_inline)) bool is_any(char p, const char* string) {
    int i = 0;
    while (string[i] != '#') {
        if (p == string[i]) {
            return true;
        }
        i++;
    }
    return false;
}

static __attribute__((always_inline)) uint32_t get_stack_data4(char** sp) {
    PSIM_WARNINGS_OFF;
    size_t alignment = (size_t)*sp % (4);
    DEBUG(printf("  alignment %ld\n", alignment));
    DEBUG(printf("  SP %p\n", (void*)*sp));
    assert(alignment == 0);
    uint32_t data = *((uint32_t*)*sp);
    *sp += 4;
    DEBUG(printf("  SP after %p\n", (void*)*sp));
    return data;
}

static __attribute__((always_inline)) uint64_t get_stack_data8(char** sp) {
    size_t alignment = (size_t)*sp % (8);
    DEBUG(printf("  alignment %ld\n", alignment));
    // realign stack pointer to "8" bytes
    *sp += alignment;
    uint64_t data = *((uint64_t*)*sp);
    *sp += 8;
    DEBUG(printf("  SP after %p\n", (void*)*sp));
    return data;
}

void __attribute__((always_inline))
my_vprintf(uint64_t* _fmt, uint64_t* _sp, uint64_t* _ret, char* buf) {
    char output[2048];
    char* out = output;
    char* in = (char*)(*_fmt);
    char* sp = (char*)(*_sp);
    uint32_t* ret = (uint32_t*)(*_ret);

    while (*in != '\0') {
        char fmt_ty[64];
        char* fmtp_ty = fmt_ty;
        if (*in == '%') {
            DEBUG(printf("---fmt-ty-start-->"));
            DEBUG(putchar(*in));
            *fmtp_ty++ = *in++;

            const char types[] = "cdiouxXpeEfgGaAs#";
            while (!is_any(*in, types) && *in != '\0' && *in != '%') {
                DEBUG(putchar(*in));
                *fmtp_ty++ = *in++;
                assert(fmtp_ty - fmt_ty < 63);
            }
            DEBUG(putchar(*in));
            *fmtp_ty++ = *in++;
            *fmtp_ty = '\0';
            DEBUG(printf("<---fmt-ty-end--\n"));
        }

        /* format token was found */
        if (fmtp_ty - fmt_ty > 1 && *(fmtp_ty - 1) != '%') {
            // find size and type:
            // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#format-specifiers
            char type = *(fmtp_ty - 1);
            assert(is_any(type, "cdiouxXpeEfgGaAs#"));
            char size[2] = {'\0', '\0'};
            char s = *(fmtp_ty - 2);
            if (s == 'h' || s == 'l') {
                size[0] = s;
                s = *(fmtp_ty - 3);
                if (s == 'l') {
                    size[1] = s;
                }
            }

            int nchars = -1;
            if (size[0] == '\0' && size[1] == '\0') {
                if (is_any(type, "idc#")) {
                    uint32_t data = get_stack_data4(&sp);
                    int val = *((int*)(&data));
                    nchars = sprintf(out, fmt_ty, val);
                    DEBUG(printf("val(d) %d\n", val));
                } else if (is_any(type, "uoxX#")) {
                    uint32_t data = get_stack_data4(&sp);
                    unsigned int val = *((unsigned int*)(&data));
                    nchars = sprintf(out, fmt_ty, val);
                    DEBUG(printf("val(x) %x\n", val));
                } else if (is_any(type, "fFeEgGaA#")) {
                    uint64_t data = get_stack_data8(&sp);
                    double val = *((double*)(&data));
                    nchars = sprintf(out, fmt_ty, val);
                    DEBUG(printf("val(f) %f\n", val));
                } else if (is_any(type, "sp#")) {
                    uint64_t data = get_stack_data8(&sp);
                    char* val = *(char**)(&data);
                    nchars = sprintf(out, fmt_ty, val);
                    DEBUG(printf(fmt_ty, val));
                } else {
                    assert(0);
                }

            } else if (size[0] == 'h' && size[1] == '\0') {
                assert(0);
            } else if (size[0] == 'l' && size[1] == '\0') {
                assert(0);
            } else if (size[0] == 'l' && size[1] == 'l') {
                if (is_any(type, "id#")) {
                    uint64_t data = get_stack_data8(&sp);
                    unsigned long long val = *(unsigned long long*)(&data);
                    nchars = sprintf(out, fmt_ty, val);
                    DEBUG(printf("val(lld) %lld\n", val));
                } else {
                    assert(0);
                }
            }
            assert(nchars > 0);

            out += nchars;
        } else if (fmtp_ty - fmt_ty > 1 && *(fmtp_ty - 1) == '%') {
            *out++ = '%';
        } else {
            DEBUG(printf("unmod %c\n", *in));
            *out++ = *in++;
        }
    }
    *out = '\0';
    *ret = sprintf(buf, "%s", output);
    fflush(stdout);
}

void __attribute__((always_inline)) print_gang(int id, char* buf) {
    uint32_t ret = 0;
    uint8_t stack[128];
    uint8_t* sp = stack;
    const char* fmt = "hello from lane %d - value %f\n";
    *((int*)sp) = id;
    sp += sizeof(int);

    // realign stack pointer to "8" bytes
    size_t alignment = (size_t)sp % (sizeof(double));
    sp += alignment;
    *((double*)sp) = id;
    sp += sizeof(double);

    uint64_t stack_val = (uint64_t)&stack;
    uint64_t ret_val = (uint64_t)&ret;
    my_vprintf((uint64_t*)&fmt, (uint64_t*)&stack_val, (uint64_t*)&ret_val,
               buf);
}

int main() {
    char buffs[GANG_SIZE][1024] = {};
#psim gang_size(GANG_SIZE)
    {
        size_t lane = psim_get_lane_num();
        print_gang(lane, buffs[lane]);
    }

    for (int i = 0; i < GANG_SIZE; i++) {
        char buf[1024] = {};
        print_gang(i, buf);
        assert(std::string(buf) == std::string(buffs[i]));
    }
    printf("Success!\n");
}
