cmake_minimum_required(VERSION 3.15...4.2)
set(CMAKE_CXX_STANDARD 23)
project(Microworld DESCRIPTION "A cellular economy game."
                LANGUAGES CXX C)

add_subdirectory(extern/pybind11)
include_directories(includes/Panda3D)
pybind11_add_module(cell src/cell_life.cpp)
