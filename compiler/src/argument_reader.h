/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#pragma once

#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "utils.h"

class ArgumentReader {
    int argc;
    char** argv;

    /* arguments are those passed as inputs */
    struct Argument {
        std::string value;
        bool checked;
    };
    std::vector<Argument> args;

    /* options are those provided by the program */
    struct Option {
        std::string name;
        std::string help;
    };
    std::vector<Option> options;

    /* set to true after help message has been checked */
    bool finalized = false;

  public:
    ArgumentReader(int _argc, char** _argv) : argc(_argc), argv(_argv) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg.empty()) {
                FATAL("argv[" << i << "] is the empty string!\n");
            }
            args.push_back({arg, false});
        }
    }

    template <class T>
    bool readOption(const std::string& name, T& oParam, std::string help = "") {
        assert(finalized == false);
        size_t verbose_pos = name.find("--v");
        if (verbose_pos != std::string::npos) {
            help = "Verbose flag for " + name.substr(3) + "(.cpp)";
        }
        addOption(name, help);
        for (size_t i = 0; i < args.size(); i++) {
            // iterate backwards so that later flags have precedence
            size_t pos = args.size() - i - 1;
            if (name == args[pos].value) {
                if (pos + 1 >= args.size()) {
                    std::cout << "Error: expected value after option " << name
                              << "\n";
                    exit(1);
                }
                oParam = get<T>(pos + 1);
                args[pos].checked = true;
                args[pos + 1].checked = true;
                return true;
            }
        }
        return false;
    }

    bool hasOption(const std::string& name, std::string help = "") {
        assert(finalized == false);
        addOption(name, help);
        for (size_t i = 0; i < args.size(); i++) {
            if (name == args[i].value) {
                args[i].checked = true;
                return true;
            }
        }
        return false;
    }

    std::string getHelpMsg() {
        finalized = true;
        std::stringstream s;
        s << "Psv (parsimony vectorizer) options:\n";
        for (auto o : options) {
            s << "    " << std::setw(16) << o.name << "    " << o.help << "\n";
        }
        return s.str();
    }

    std::string finalize() {
        std::string unused_args = getUnusedArguments();
        if (!unused_args.empty()) {
            return "Unexpected arguments: " + unused_args + "\n";
        }
        finalized = true;
        return unused_args;
    }

  private:
    std::string getUnusedArguments() {
        std::string s;
        for (auto& i : args) {
            if (!i.checked) {
                if (!s.empty()) {
                    s += "; ";
                }
                s += "'" + i.value + "'";
            }
        }
        return s;
    }

    template <class T>
    T get(int idx) {
        std::istringstream ss(args[idx].value);
        T result;
        ss >> result;
        return result;
    }

    bool addOption(std::string name, std::string help) {
        for (auto o : options) {
            /* option already present */
            if (name == o.name) {
                return false;
            }
        }
        options.push_back({name, help});
        return true;
    }
};
