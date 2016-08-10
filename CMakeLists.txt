cmake_minimum_required(VERSION 3.3)
project(liblinear_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


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
    train.c
    tron.cpp
    tron.h
    Solver_SOFTMAX.cpp
    Solver_SOFTMAX.h
    Solver_EG.cpp
    Solver_EG.h
    Solver_ME_DUAL.h
    Solver_ME_DUAL.cpp
    Solver_NEW.h
    Solver_NEW.cpp
        Function_SOFTMAX.cpp Function_SOFTMAX.h)

add_executable(train ${SOURCE_FILES})

set(SOURCE_FILES
    blas/blas.h
    blas/blasp.h
    blas/daxpy.c
    blas/ddot.c
    blas/dnrm2.c
    blas/dscal.c
    linear.cpp
    linear.h
    predict.c
    #train.c
    tron.cpp
    tron.h
    Solver_SOFTMAX.cpp
    Solver_SOFTMAX.h
    Solver_EG.cpp
    Solver_EG.h
    Solver_ME_DUAL.h
    Solver_ME_DUAL.cpp
    Solver_NEW.h
    Solver_NEW.cpp
        Function_SOFTMAX.cpp Function_SOFTMAX.h)


add_executable(predict ${SOURCE_FILES})

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
    main.cpp
    tron.cpp
    tron.h
    Solver_SOFTMAX.cpp
    Solver_SOFTMAX.h
    Solver_EG.cpp
    Solver_EG.h
    Solver_ME_DUAL.h
    Solver_ME_DUAL.cpp
    Solver_NEW.h
    Solver_NEW.cpp
        Function_SOFTMAX.cpp Function_SOFTMAX.h)


add_executable(main ${SOURCE_FILES})