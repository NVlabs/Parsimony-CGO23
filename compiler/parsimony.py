#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import subprocess
import re
import argparse
import os
import tempfile
import time

script_path = os.path.dirname(os.path.realpath(__file__))
llvm_path = "${LLVM_INSTALL_DIR}";
llvm_backend_path = "${LLVM_BACKEND_DIR}";
sleef_path = "${SLEEF_INSTALL_DIR}";
if not llvm_backend_path:
    llvm_backend_path = llvm_path

###########################################################################################################

def run(args, cmd):
    begin = time.time()
    cmd = re.sub(" +", " ", cmd).strip()  # eliminate extra spaces
    if args.verbose:
        sys.stderr.write("Command is: " + cmd + "\n")
    result = subprocess.run(cmd.split(" "), stderr=subprocess.PIPE,
            stdout=subprocess.PIPE)
    end = time.time()
    sys.stdout.write(result.stdout.decode('utf-8',errors="ignore"))
    sys.stderr.write(result.stderr.decode('utf-8',errors="ignore"))
    if args.verbose:
        sys.stderr.write("Command took " + str(end - begin) + " seconds\n")
    if result.returncode != 0 or args.verbose > 1:
        sys.stderr.write("Exit with error code " + str(result.returncode) + "\n")
        sys.stderr.write("Command was: " + cmd.strip() + "\n")
    if result.returncode != 0:
        sys.exit(result.returncode)

###########################################################################################################

def brace_depth(s, lbrace, rbrace, depth = 0):
    content = ""
    nocontent = ""
    for c in s:
        prev_depth = depth
        if c == lbrace:
            depth += 1
        elif c == rbrace:
            depth -= 1
            if depth < 0:
                raise Exception
        if depth != 0 or prev_depth == 1:
            content += c
        else:
            nocontent += c
        if depth == 0 and prev_depth == 1:
            break
    return depth, content, nocontent

###########################################################################################################

def genParReg(name, body, has_cond, linemarker):
    s  = "int __attribute__((annotate(\"fence\"))) __psim_fence_attr;\n"
    s += "(void) __psim_fence_attr;\n"
    s += "__psim_set_gang_size((unsigned) __psim_gang_size);\n"
    s += "__psim_set_gang_num(__psim_i/__psim_gang_size);\n"
    s += "__psim_set_grid_size(__psim_grid_size);\n"
    s += "__psim_set_grid_sub_name(\"" + name + "\");\n"
    s += "#pragma omp parallel\n"
    s += "{\n"
    if has_cond:
        s += "    if(__psim_i + psim_get_lane_num() < __psim_grid_size)\n"
    s += linemarker
    s += "    " + body +"\n"
    s += "}\n"
    return s

###########################################################################################################

launch_gangs_head_body_tail_template = """
    {
        uint64_t __psim_i = 0;
        const uint64_t __psim_grid_size = $GRID_SIZE$;
        const uint64_t __psim_gang_size = $GANG_SIZE$;
        $PARALLEL$
        for(__psim_i = 0; __psim_i < __psim_grid_size; __psim_i += $GANG_SIZE$) {
            if ( __psim_i  == 0 && $GANG_SIZE$ == __psim_grid_size) {
                $HEAD_TAIL$
            } else if (__psim_i  == 0) {
                $HEAD$
            } else if (__psim_i + $GANG_SIZE$ < __psim_grid_size) {
                $BODY$
            } else if (__psim_i + $GANG_SIZE$ == __psim_grid_size) {
                $TAIL$
            }
        }
    }
"""

launch_gangs_body_tail_template = """
    {
        uint64_t __psim_i = 0;
        const uint64_t __psim_grid_size = $GRID_SIZE$;
        const uint64_t __psim_gang_size = $GANG_SIZE$;
        $PARALLEL$
        for(__psim_i = 0; __psim_i < __psim_grid_size; __psim_i += $GANG_SIZE$) {
            if (__psim_i + $GANG_SIZE$ == __psim_grid_size) {
                $TAIL$
            } else {
                $BODY$
            }
        }
    }
"""

launch_gangs_head_body_template = """
    {
        uint64_t __psim_i = 0;
        const uint64_t __psim_grid_size = $GRID_SIZE$;
        const uint64_t __psim_gang_size = $GANG_SIZE$;
        $PARALLEL$
        for(__psim_i = 0; __psim_i < __psim_grid_size; __psim_i += $GANG_SIZE$) {
            if (__psim_i  == 0) {
                $HEAD$
            } else {
                $BODY$
            }
        }
    }
"""

launch_gangs_body_template = """
    {
        uint64_t __psim_i = 0;
        const uint64_t __psim_grid_size = $GRID_SIZE$;
        const uint64_t __psim_gang_size = $GANG_SIZE$;
        $PARALLEL$
        for(__psim_i = 0; __psim_i < __psim_grid_size; __psim_i += $GANG_SIZE$) {
            $BODY$
        }
    }
"""
###########################################################################################################

launch_threads_head_body_tail_template = """
    {
        uint64_t __psim_i = 0;
        const uint64_t __psim_grid_size = $GRID_SIZE$;
        const uint64_t __psim_gang_size = $GANG_SIZE$;
        $PARALLEL$
        for(__psim_i = 0; __psim_i < __psim_grid_size; __psim_i += $GANG_SIZE$) {
            if ( __psim_i  == 0 && $GANG_SIZE$ == __psim_grid_size) {
                $HEAD_TAIL$
            } else if (__psim_i  == 0 && $GANG_SIZE$ > __psim_grid_size) {
                $HEAD_TAIL_COND$
            } else if (__psim_i  == 0) {
                $HEAD$
            } else if (__psim_i + $GANG_SIZE$ < __psim_grid_size) {
                $BODY$
            } else if (__psim_i + $GANG_SIZE$ == __psim_grid_size) {
                $TAIL$
            } else {
                $TAIL_COND$
            }
        }
    }
"""

launch_threads_body_tail_template = """
    {
        uint64_t __psim_i = 0;
        const uint64_t __psim_grid_size = $GRID_SIZE$;
        const uint64_t __psim_gang_size = $GANG_SIZE$;
        $PARALLEL$
        for(__psim_i = 0; __psim_i < __psim_grid_size; __psim_i += $GANG_SIZE$) {
            if (__psim_i  == 0 && $GANG_SIZE$ > __psim_grid_size) {
                $TAIL_COND$
            } else if (__psim_i + $GANG_SIZE$ < __psim_grid_size) {
                $BODY$
            } else if (__psim_i + $GANG_SIZE$ == __psim_grid_size) {
                $TAIL$
            }
        }
    }
"""

launch_threads_head_body_template = """
    {
        uint64_t __psim_i = 0;
        const uint64_t __psim_grid_size = $GRID_SIZE$;
        const uint64_t __psim_gang_size = $GANG_SIZE$;
        $PARALLEL$
        for(__psim_i = 0; __psim_i < __psim_grid_size; __psim_i += $GANG_SIZE$) {
            if (__psim_i  == 0 && $GANG_SIZE$ > __psim_grid_size) {
                $HEAD_COND$
            } else if (__psim_i  == 0) {
                $HEAD$
            } else if (__psim_i + $GANG_SIZE$ <= __psim_grid_size) {
                $BODY$
            } else {
                $BODY_COND$
            }
        }
    }
"""

launch_threads_body_template = """
    {
        uint64_t __psim_i = 0;
        const uint64_t __psim_grid_size = $GRID_SIZE$;
        const uint64_t __psim_gang_size = $GANG_SIZE$;
        $PARALLEL$
        for(__psim_i = 0; __psim_i < __psim_grid_size; __psim_i += $GANG_SIZE$) {
            if (__psim_i + $GANG_SIZE$ <= __psim_grid_size) {
                $BODY$
            } else {
                $BODY_COND$
            }
        }
    }
"""
###########################################################################################################
# true or false if the have some value
known_directives = { "gang_size": True,
                     "num_spmd_threads": True,
                     "num_spmd_gangs": True,
                     "parallel": False}

def process_psim_annotations(infilename, outfilename, args):
    if args.verbose:
        sys.stderr.write("process_psim_annotations " + infilename + " " + outfilename + "\n")
    psim_pragma = "#psim"
    omp_psim_pragma = "#pragma omp psim"
    outcode = ""
    with open(infilename, "r") as f:
        outcode = f.read()
        outcode = outcode.replace(psim_pragma, omp_psim_pragma)

    # write processed file
    with open(outfilename, "w") as f:
        f.write(outcode)


def process_omp_psim_pragmas(infilename, outfilename, orig_filename, args):
    if args.verbose:
        sys.stderr.write("process_omp_psim_pragmas " + infilename + " " + outfilename + "\n")
    psim_pragma = "#pragma omp psim"
    outcode = ""
    with open(infilename, "r") as f:
        l = "true"
        line_count = 0
        while l:
            #read next line
            l = f.readline()

            if l.startswith("#") and orig_filename in l:
                line_count = int(l.split()[1]) - 1
            line_count += 1

            #(remove extra spaces)
            lp = " ".join(l.split())
            #look only for lines that contain our pragma
            if lp.startswith(psim_pragma):
                directives = {}
                pos = len(psim_pragma)
                while pos < len(lp):
                    depth, value, names = brace_depth(l[pos:],"(", ")")
                    assert(depth == 0)
                    pos += len(names) + len(value)
                    names = names.split();
                    # empty directives  (i.e. "parallel" in  "parallel num_spmd_threads(size)")
                    for n in names[:-1]:
                        n = n.rstrip().lstrip()
                        directives[n] = ""
                    # last one is the directive with a value
                    n = names[-1].rstrip().lstrip()
                    directives[n] = value.rstrip().lstrip()

                for d, v in directives.items():
                    if d not in known_directives:
                        sys.stderr.write("parsimony: error: \"#psim\" directive: " + d + " unknown!\n\n")
                        sys.exit(1)

                num_spmd_threads = directives.get("num_spmd_threads")
                num_spmd_gangs = directives.get("num_spmd_gangs")
                gang_size = directives.get("gang_size")
                if not gang_size:
                    sys.stderr.write("parsimony: error: \"#psim\" must specify gang_size!\n\n")
                    sys.exit(1)
                parallel = ""
                if "parallel" in directives:
                    parallel = "#pragma omp parallel for"

                if num_spmd_gangs and num_spmd_threads:
                    sys.stderr.write("parsimony: error: \"#psim\" cant specify num_spmd_gangs and num_spmd_threads at the same time\n\n")
                    sys.exit(1)
                elif not num_spmd_gangs and not num_spmd_threads:
                    num_spmd_gangs = "1"


                depth = 0
                body = ""
                body_line_start = line_count
                while l:
                    l = f.readline()
                    if l.startswith("#") and orig_filename in l:
                        line_count = int(l.split()[1]) - 1
                    line_count += 1


                    depth, content, nocontent = brace_depth(l, "{", "}", depth)
                    if nocontent.rstrip():
                        sys.stderr.write("parsimony: error: \"#psim\" must be scoped within { } found instead:\n\n" + nocontent + "\n\n")
                        sys.exit(1)
                    body += content
                    if body.rstrip() and depth == 0:
                        break

                assert(body)
                uses_tail_gang = body.find("psim_is_tail_gang") != -1
                uses_head_gang = body.find("psim_is_head_gang") != -1

                if args.verbose:
                    sys.stderr.write("Found #psim\n")
                    sys.stderr.write("num_spmd_threads: " + str(num_spmd_threads) + "\n")
                    sys.stderr.write("num_spmd_gangs: " + str(num_spmd_gangs) + "\n")
                    sys.stderr.write("gang_size: " + str(gang_size) + "\n")
                    sys.stderr.write("uses psim_is_tail_gang(): " + str(uses_tail_gang) + "\n")
                    sys.stderr.write("uses psim_is_head_gang(): " + str(uses_head_gang) + "\n")
                    sys.stderr.write("parallel: " + parallel + "\n")

                body_gang = body.replace("psim_is_tail_gang()", "false").replace("psim_is_head_gang()", "false")
                tail_gang = body.replace("psim_is_tail_gang()", "true").replace("psim_is_head_gang()", "false")
                head_gang = body.replace("psim_is_tail_gang()", "false").replace("psim_is_head_gang()", "true")
                head_tail_gang = body.replace("psim_is_tail_gang()", "true").replace("psim_is_head_gang()", "true")


                if num_spmd_gangs:
                    if uses_tail_gang and uses_head_gang:
                        launch = launch_gangs_head_body_tail_template
                    elif not uses_tail_gang and uses_head_gang:
                        launch = launch_gangs_head_body_template
                    elif uses_tail_gang and not uses_head_gang:
                        launch = launch_gangs_body_tail_template
                    else:
                        launch = launch_gangs_body_template
                    grid_size = "((" + num_spmd_gangs + ") * (" + gang_size + "))"
                else:
                    assert num_spmd_threads
                    if uses_tail_gang and uses_head_gang:
                        launch = launch_threads_head_body_tail_template
                    elif not uses_tail_gang and uses_head_gang:
                        launch = launch_threads_head_body_template
                    elif uses_tail_gang and not uses_head_gang:
                        launch = launch_threads_body_tail_template
                    else:
                        launch = launch_threads_body_template
                    grid_size = num_spmd_threads


                linemarker = "# " +  str(body_line_start) + " \"" + orig_filename + "\"\n"

                launch = launch.replace("$PARALLEL$", parallel)
                launch = launch.replace("$GANG_SIZE$", gang_size)
                launch = launch.replace("$GRID_SIZE$", grid_size)

                launch = launch.replace("$HEAD_TAIL$", genParReg("head_tail_gang", head_tail_gang, False, linemarker))
                launch = launch.replace("$TAIL$", genParReg("tail_gang", tail_gang, False, linemarker))
                launch = launch.replace("$HEAD$", genParReg("head_gang", head_gang, False, linemarker))
                launch = launch.replace("$BODY$", genParReg("body_gang", body_gang, False, linemarker))

                launch = launch.replace("$HEAD_TAIL_COND$", genParReg("head_tail_gang_bound_check", head_tail_gang, True, linemarker))
                launch = launch.replace("$TAIL_COND$", genParReg("tail_gang_bound_check", tail_gang, True, linemarker))
                launch = launch.replace("$HEAD_COND$", genParReg("head_gang_bound_check", head_gang, True, linemarker))
                launch = launch.replace("$BODY_COND$", genParReg("body_gang_bound_check", body_gang, True, linemarker))

                launch += "# " +  str(line_count) + " \"" + orig_filename + "\"\n"
                outcode += launch

            else:
                # just a standard line
                outcode += l

        # write processed file
        with open(outfilename, "w") as f:
            f.write(outcode)





def run_compiler_steps(infilename, args, unknownargs):
    with tempfile.TemporaryDirectory() as d:
        if args.tmpdir:
            d = args.tmpdir

        try:
            os.mkdir(d)
        except FileExistsError:
            pass

        tmp_filename_base = d + os.sep + os.path.basename(infilename)

        #step 0: front-end -- replace #psim with #pragma omp psim
        preproc0_file = os.path.basename(infilename) + ".pp0.cpp"
        process_psim_annotations(infilename, preproc0_file, args)

        #step 1: front-end -- run clang preprocessor
        preproc1_file = tmp_filename_base + ".pp1.cpp"
        run(args, llvm_path + "/bin/clang++ -E -fopenmp " + unknownargs + \
                " -isystem " + script_path + "/../include " + \
                " " + preproc0_file + \
                " -o " + preproc1_file)
        os.rename(preproc0_file, tmp_filename_base + ".pp0.cpp")
        
        #step 2: front-end -- process #pragma omp psim by leveraging #pragma omp parallel for
        preproc2_file = tmp_filename_base + ".pp2.cpp"
        process_omp_psim_pragmas(preproc1_file, preproc2_file, infilename, args)


        #step 3: front-end -- compile file in pre-bitcode
        pre_vec_bitcode_file = tmp_filename_base + ".pre_vec.ll"
        run(args, llvm_path + "/bin/clang++ -fopenmp " + unknownargs + \
                " -Xclang -no-opaque-pointers " + \
                " -fno-vectorize -fno-slp-vectorize -fno-unroll-loops " + \
                " -isystem " + script_path + "/../include " + \
                " -S -emit-llvm -g1 -c " +  preproc2_file + \
                " -o " + pre_vec_bitcode_file)

        #step 4: middle-end -- call psv to vectorize pre-bitcode into post-bitcode
        post_vec_bitcode_file = tmp_filename_base + ".post_vec.ll"
        run(args, script_path + "/psv -i " + \
                pre_vec_bitcode_file + " -o " + \
                post_vec_bitcode_file + " " + args.extra_psv_args)

        # step 5: back-end -- compile to object or binary
        if args.compile:
            if args.outputfile:
                final_outfilename = args.outputfile
            else:
                final_outfilename = infilename + ".o"
        else:
            final_outfilename = tmp_filename_base + ".o"

        run(args, llvm_backend_path + "/bin/clang++ -fopenmp " + unknownargs + " -Wno-unused-command-line-argument -c " + \
                post_vec_bitcode_file + " -o " + final_outfilename )
        return final_outfilename

def main():
    argparser = argparse.ArgumentParser("parsimony", conflict_handler="resolve")

    # intercept compiler options
    argparser.add_argument("inputfiles", type=str, nargs='*')
    argparser.add_argument("-o", dest="outputfile", type=str)
    argparser.add_argument("-c", dest="compile", action="store_const", const="-c", default="")

    argparser.add_argument("-g", dest="debug", action="store_true")

    # script options they all start with --X
    argparser.add_argument("--Xpsv", dest="extra_psv_args", type=str, default="", help="Extra argument passed to psv.")
    argparser.add_argument("--Xtmp", dest="tmpdir", type=str, default="tmp", help="Folder for temporary files.")
    argparser.add_argument("--Xv",   dest="verbose", action="store_true", help="Verbose flag for the parsimony script.")
    argparser.add_argument("-h",     dest="help", action="store_true", help="Print help message.")

    args, unknownargs = argparser.parse_known_args()

    if args.verbose:
        sys.stderr.write("Args: " + " ".join(sys.argv) + "\n")

    if not args.inputfiles or args.help:
        argparser.print_help()
        sys.stderr.write(80 * "-" + "\n")
        run(args, script_path + "/psv -h ")
        if not args.help:
            sys.exit(1)
        else:
            sys.exit(0)

    if args.debug:
        sys.stderr.write("parsimony: ignoring -g for now\n")

    if len(args.inputfiles) > 1 and args.outputfile and args.compile:
        sys.stderr.write("parsimony: error: cannot specify -o when generating multiple output files\n")
        sys.exit(1)

    if not args.outputfile and not args.compile:
        args.outputfile = "a.out"

    objs = []
    for f in args.inputfiles:
        if f.endswith(".cpp") or f.endswith(".cxx") or f.endswith(".c") or f.endswith(".cc"):
            objs.append(run_compiler_steps(f, args, " ".join(unknownargs)))
        elif f.endswith(".o"):
            objs.append(f)
        else:
            std.stderr.write("parsimony: error: unknown file extension " + f + "\n")
            sys.exit(1)

    # link step
    if not args.compile:
        if not sleef_path:
            run(args, llvm_path + "/bin/clang++ -fopenmp " + " ".join(unknownargs) + \
                " -Wl,-rpath," + llvm_path + "/lib/ " +  " ".join(objs)  + " -o " + args.outputfile)
        else:
            run(args, llvm_path + "/bin/clang++ -fopenmp " + " ".join(unknownargs) + \
                " -Wl,-rpath," + sleef_path + "/lib64/ " +  " -Wl,-rpath," + llvm_path + \
                "/lib/ " +  " ".join(objs) + " -L" + sleef_path + "/lib64 -lsleef" + " -o " + args.outputfile)

    if args.verbose:
        sys.stderr.write("Done!\n")

if __name__ == "__main__":
    main()
