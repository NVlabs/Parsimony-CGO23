/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <cassert>
#include <iostream>
#include "shape.h"
#include "shape_calc.h"

using namespace ps;

template <typename T, typename... S>
void check(z3::context& ctx, T t, Shape sa, S... other_shapes) {
    unsigned width = sa.base.get_sort().bv_size();

    std::cout << "Checking " << t.name << " with " << width << " bits\n";

    z3::expr actual =
        t.f_expr(sa.getExprAtLane(0), other_shapes.getExprAtLane(0)...);

    z3::expr base = t.f_expr(sa.base, other_shapes.base...);
    z3::expr index = t.f_proposed_index(0, sa, other_shapes...);
    z3::expr proposed = base + index;

    z3::solver s(ctx);

    // Add assumptions
    for (auto f : t.assumptions) {
        s.add(f(sa, other_shapes...));
    }

    // Look for a counterexample
    s.add(proposed != actual);

    // Run the solver
    switch (s.check()) {
        case z3::sat: {
            std::cout << "Found counterexample!\n";
            z3::model m = s.get_model();
            for (Shape i : {sa, other_shapes...}) {
                std::cout << i.toString(true) << "\n";
                std::cout << "Base:  " << m.eval(i.base, true).to_string()
                          << "\n";
                std::cout << "Index: " << m.eval(i.indices[0], true).to_string()
                          << "\n";
            }
            for (z3::expr i : {base, index, proposed, actual}) {
                std::cout << i.simplify().to_string() << " = "
                          << m.eval(i, true).to_string() << "\n";
            }
            fflush(stdout);
            assert(false);
        } break;
        case z3::unknown:
            std::cout << "Solver returned unknown!\n";
            fflush(stdout);
            assert(false);
            break;
        case z3::unsat:
            std::cout << "No counterexamples!\n";
            break;
        default:
            assert(false);
    }
}

template <typename T, typename... S>
void checkRedundancy(z3::context& ctx, T t1, T t2, Shape sa,
                     S... other_shapes) {
    unsigned width = sa.base.get_sort().bv_size();

    std::cout << "Checking " << t1.name << " vs. " << t2.name << " with "
              << width << " bits\n";

    // Add assumptions
    z3::solver s(ctx);
    for (auto f : t1.assumptions) {
        s.add(f(sa, other_shapes...));
    }
    z3::expr_vector v(sa.base.ctx());
    for (auto f : t2.assumptions) {
        v.push_back(!f(sa, other_shapes...));
    }
    s.add(z3::mk_or(v));

    // Run the solver
    switch (s.check()) {
        case z3::sat: {
            std::cout << "Found counterexample!\n";
            z3::model m = s.get_model();
            for (Shape i : {sa, other_shapes...}) {
                std::cout << i.eval(m).toString() << "\n";
            }
        } break;
        case z3::unknown:
            std::cout << "Solver returned unknown!\n";
            fflush(stdout);
            assert(false);
            break;
        case z3::unsat:
            std::cout << "No counterexamples...condition is redundant!!\n";
            fflush(stdout);
            assert(false);
            break;
        default:
            assert(false);
    }
}

int main(int argc, char* argv[]) {
    unsigned bit_counts[] = {3, 4, 8, 10};  // 16, 32, 64};

    for (unsigned num_bits : bit_counts) {
        z3::context ctx;

        Shape a =
            Shape::Indexed(Shape::symbolicExpr(ctx, "a_base", num_bits),
                           {Shape::symbolicExpr(ctx, "a_index", num_bits)});
        Shape b =
            Shape::Indexed(Shape::symbolicExpr(ctx, "b_base", num_bits),
                           {Shape::symbolicExpr(ctx, "b_index", num_bits)});

        if (argc > 1 && std::string(argv[1]) == "-r") {
            std::vector<std::string> transforms_to_check{"and1", "and2", "and3",
                                                         "and4"};
            for (auto i : transforms_to_check) {
                for (auto j : transforms_to_check) {
                    if (i == j) {
                        continue;
                    }
                    checkRedundancy(ctx, known_transforms.binary[i],
                                    known_transforms.binary[j], a, b);
                }
            }
        } else {
            check(ctx, known_transforms.sext(num_bits * 2), a);
            check(ctx, known_transforms.trunc(num_bits / 2), a);
            check(ctx, known_transforms.zext(num_bits * 2), a);

            for (auto t : known_transforms.binary) {
                check(ctx, t.second, a, b);
            }
        }
    }

    return 0;
}
