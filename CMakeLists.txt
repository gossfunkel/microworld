cmake_minimum_required(VERSION 3.15...4.2)
set(CMAKE_CPP_COMPILER MSVC)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
project(cell_life LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /LD /O2 /std:c++latest /Wall /await /favor:INTEL64")

include_directories(C:/Panda3D-1.11.0-x64/include)
link_directories(C:/Panda3D-1.11.0-x64/lib)
add_library(cell_life SHARED src/cell_life.cpp)
target_link_libraries(cell_life
    libp3framework
    libpanda
    libpandafx
    libpandaexpress
    libp3dtoolconfig
    libp3dtool
    libp3direct
)