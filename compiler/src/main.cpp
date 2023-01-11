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
#include <sstream>
#include <unordered_set>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include "argument_reader.h"
#include "diagnostics.h"
#include "function.h"
#include "inst_order.h"
#include "live_out.h"
#include "mask.h"
#include "module.h"
#include "prints.h"
#include "shapes.h"
#include "transform.h"
#include "utils.h"

using namespace llvm;
using namespace ps;

Module* createModuleFromFile(const std::string& fileName,
                             LLVMContext& context) {
    SMDiagnostic diag;
    auto modPtr = llvm::parseIRFile(fileName, diag, context);
    return modPtr.release();
}

int main(int argc, char** argv) {
    ArgumentReader reader(argc, argv);

    std::string inFile, outFile;
    bool hasInFile =
        reader.readOption<std::string>("-i", inFile, "Input llvm file");
    bool hasOutFile =
        reader.readOption<std::string>("-o", outFile, "Output llvm file");
    global_opts.add_prints =
        reader.hasOption("-p",
                         "Adds print statement after each llvm vectorized "
                         "instruction (for debug purposes)");
    global_opts.scalable_size = 0;
    reader.readOption<int>(
        "-S", ps::global_opts.scalable_size,
        "SVE scalable size (0=fixed-size (non scalable), 1=128bit SVE, "
        "2=256bit SVE, 4=512bit SVE)");
    global_opts.error_on_warn =
        reader.hasOption("-Werror", "Treat the warnings as errors");
    global_opts.ignore_warn_set = reader.hasOption(
        "-Iwarnset", "Ignore set of warning on/off inside the application");

    unsigned verbosity_level = 0;
    reader.readOption<unsigned>("-v", verbosity_level, "Global verbosity flag");
    broadcast_verbosity_level = verbosity_level;
    diagnostics_verbosity_level = verbosity_level;
    function_verbosity_level = verbosity_level;
    inst_order_verbosity_level = verbosity_level;
    live_out_verbosity_level = verbosity_level;
    mask_verbosity_level = verbosity_level;
    module_verbosity_level = verbosity_level;
    prints_verbosity_level = verbosity_level;
    resolver_verbosity_level = verbosity_level;
    shapes_verbosity_level = verbosity_level;
    transform_verbosity_level = verbosity_level;
    vectorize_verbosity_level = verbosity_level;
    value_cache_verbosity_level = verbosity_level;
    vfabi_verbosity_level = verbosity_level;

    reader.readOption<unsigned>("--vbroadcast", broadcast_verbosity_level);
    reader.readOption<unsigned>("--vdiagnostics", diagnostics_verbosity_level);
    reader.readOption<unsigned>("--vfunction", function_verbosity_level);
    reader.readOption<unsigned>("--vinst_order", inst_order_verbosity_level);
    reader.readOption<unsigned>("--vlive_out", live_out_verbosity_level);
    reader.readOption<unsigned>("--vmask", mask_verbosity_level);
    reader.readOption<unsigned>("--vmodule", module_verbosity_level);
    reader.readOption<unsigned>("--vprints", prints_verbosity_level);
    reader.readOption<unsigned>("--vresolver", resolver_verbosity_level);
    reader.readOption<unsigned>("--vshapes", shapes_verbosity_level);
    reader.readOption<unsigned>("--vtransform", transform_verbosity_level);
    reader.readOption<unsigned>("--vvectorize", vectorize_verbosity_level);
    reader.readOption<unsigned>("--vvalue_cache", value_cache_verbosity_level);
    reader.readOption<unsigned>("--vvfabi", vfabi_verbosity_level);

    // no more reader calls to collect new variables after this
    if (reader.hasOption("-h", "Help")) {
        std::cout << reader.getHelpMsg();
        return 0;
    }

    std::string msg = reader.finalize();
    if (!msg.empty()) {
        std::cerr << msg << "\n";
        return 1;
    }

    if (!hasInFile) {
        std::cerr << "No input file specified\n";
        reader.getHelpMsg();
        return 1;
    }

    LLVMContext context;

    // Load module
    llvm::Module* mod = createModuleFromFile(inFile, context);
    if (!mod) {
        FATAL("Could not load module " << inFile << ". Aborting!\n");
        return 1;
    }

    bool broken = verifyModule(*mod, &errs());
    if (broken) {
        FATAL("Broken module!\n");
        return 1;
    }

    // Vectorize
    VectorizedModuleInfo vm_info(mod);
    ModuleVectorizer module_vectorizer(vm_info);
    module_vectorizer.initialize();
    if (hasOutFile) {
        // FIXME don't hard-code .ll
        module_vectorizer.writeToFile(outFile + ".afterPreprocess.ll");
    }

    module_vectorizer.vectorizeFunctions();

    if (hasOutFile) {
        module_vectorizer.writeToFile(outFile);
        PRINT_LOW("Final module written to \"" << outFile << "\"\n");
    } else {
        mod->print(llvm::outs(), nullptr, false, true);
    }

    return 0;
}
