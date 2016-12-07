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
    Solver_ADMM.h
    Solver_ADMM.cpp
    Solver_ADMM2.h
    Solver_ADMM2.cpp
    Solver_ALM.h
    Solver_ALM.cpp
    Solver_ALM_FW.h
    Solver_ALM_FW.cpp
    Solver_FW.h
    Solver_FW.cpp
    Solver_BLG.h
    Solver_BLG.cpp
    Function_SOFTMAX.cpp
    Function_SOFTMAX.h
    BLG/driver.h
    BLG/driver.cpp
    BLG/BLG.h
        BLG/BLG.cpp
    BLG/BLG_user.h)

add_executable(train train.c ${SOURCE_FILES})

add_executable(predict predict.c ${SOURCE_FILES})

add_executable(main main.cpp ${SOURCE_FILES})


