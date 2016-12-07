//
//

#ifndef LIBLINEAR_TEST_DRIVER_H
#define LIBLINEAR_TEST_DRIVER_H

#include <math.h>
#include "BLG.h"

void myvalue
        (
                BLGobjective *user
        ) ;

void mygrad
        (
                BLGobjective *use
        ) ;

void myvalgrad
        (
                BLGobjective *user
        ) ;

//int BLG_main (int n, double *x, double *lo, double *hi, double *a, double b);
int BLG_main (int n, double *x, double *lo, double *hi, double *a, double b, \
             double norm, double *q, double *alpha_list, double C, int yi);

#endif //LIBLINEAR_TEST_DRIVER_H
