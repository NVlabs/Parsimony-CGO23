cmake_minimum_required(VERSION 3.13.4)
project(psv)

set(LLVM_INSTALL_DIR "" CACHE PATH "LLVM install directory")
set(LLVM_BACKEND_DIR "" CACHE PATH "LLVM backend directory")

list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/llvm")
find_package(LLVM 15 REQUIRED CLANG PATHS ${LLVM_INSTALL_DIR}/lib/cmake/llvm)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM "${Z3_INSTALL_DIR}/include")

if(SLEEF_INSTALL_DIR)
    add_definitions(-DSLEEF_ENABLE=1)
endif()

set(CMAKE_CXX_STANDARD 17)

if(NOT LLVM_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(psv
    src/argument_reader.h
    src/broadcast.cpp
    src/broadcast.h
    src/diagnostics.cpp
    src/diagnostics.h
    src/function.cpp
    src/function.h
    src/main.cpp
    src/mask.cpp
    src/mask.h
    src/module.cpp
    src/module.h
    src/inst_order.cpp
    src/inst_order.h
    src/live_out.cpp
    src/live_out.h
    src/prints.cpp
    src/prints.h
    src/rename_values.cpp
    src/rename_values.h
    src/resolver.cpp
    src/resolver.h
    src/shapes.cpp
    src/shapes.h
    src/shape.cpp
    src/shape.h
    src/shape_calc.cpp
    src/shape_calc.h
    src/transform.cpp
    src/transform.h
    src/utils.cpp
    src/utils.h
    src/value_cache.cpp
    src/value_cache.h
    src/vectorize.cpp
    src/vectorize.h
    src/vfabi.cpp
    src/vfabi.h
)
add_executable(shape_checker
    src/argument_reader.h
    src/shape.h
    src/shape.cpp
    src/shape_calc.cpp
    src/shape_calc.h
    src/shape_checker.cpp
    src/utils.cpp
    src/utils.h
)

include_directories("${CMAKE_SOURCE_DIR}/src")

string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

llvm_map_components_to_libnames(llvm_libs support core irreader analysis scalaropts passes transformutils demangle)
target_link_libraries(psv ${llvm_libs})
target_link_libraries(psv "${Z3_INSTALL_DIR}/lib/libz3.so")

# shape_checker shouldn't really need llvm_libs, but utils adds that dependency...
target_link_libraries(shape_checker ${llvm_libs})
target_link_libraries(shape_checker "${Z3_INSTALL_DIR}/lib/libz3.so")

install(TARGETS psv DESTINATION bin)
install(TARGETS shape_checker DESTINATION bin)

configure_file("${CMAKE_SOURCE_DIR}/parsimony.py" "${CMAKE_CURRENT_BINARY_DIR}/parsimony")
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/parsimony" DESTINATION bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include FILES_MATCHING PATTERN "*.h*")
