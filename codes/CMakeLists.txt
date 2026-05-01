cmake_minimum_required(VERSION 3.15)
project(cat_geometry_gmsh)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(GMSH_INCLUDE_DIR gmsh.h)
find_library(GMSH_LIBRARY gmsh)

add_executable(cat_geom cat_geometry.cpp)
target_include_directories(cat_geom PRIVATE ${GMSH_INCLUDE_DIR})
target_link_libraries(cat_geom PRIVATE ${GMSH_LIBRARY})
