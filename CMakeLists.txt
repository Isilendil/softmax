cmake_minimum_required(VERSION 3.3)
project(liblinear_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0")


set(SOURCE_FILES
    blas/blas.h
    blas/blasp.h
    blas/daxpy.c
    blas/ddot.c
    blas/dnrm2.c
    blas/dscal.c
    linear.cpp
    linear.h
    #predict.c
    #train.c
    tron.cpp
    tron.h
    Solver_SGD.cpp
    Solver_SGD.h
    Solver_EG.cpp
    Solver_EG.h
    Solver_CD_DUAL.h
    Solver_CD_DUAL.cpp
    Solver_NEW.h
    Solver_NEW.cpp
    Function_SOFTMAX.cpp
    Function_SOFTMAX.h)

add_executable(train train.c ${SOURCE_FILES})

add_executable(predict predict.c ${SOURCE_FILES})

add_executable(main main.cpp ${SOURCE_FILES})


