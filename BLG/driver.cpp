/* This example solves the problem

        minimize sum_{i=1}^n exp (x [i]) - sqrt(i+1)*x [i]
        subject to a'x = b, lo <= x <= hi
        where a_i = 1 for i <= n1 or i > n2
              a_i =-1 for n1 < i <= n2
              lo_i = 0 and hi_i = 2 for i <= n1
              lo_i = 1 and hi_i = 3 for n1 < i <= n2
              lo_i = 2 and hi_i = 4 for i > n2
   The output on a linux workstation was the following:

------------------------------------------------------
    Convergence tolerance for gradient satisfied
    
    KKT error:          7.321921e-13
    Feasibility error:  5.684342e-14
    function value:    -6.470662e+02
    
    Iterations:           42
    Function evaluations: 49
    Gradient evaluations: 42
    Mu computation:       0
    
    USE affine scaling search direction
    
    Convergence tolerance for gradient satisfied
    
    KKT error:          3.907985e-13
    Feasibility error:  4.263256e-14
    function value:    -6.470662e+02
    
    Iterations:           54
    Function evaluations: 70
    Gradient evaluations: 56
    Mu computation:       107
------------------------------------------------------

To solve the problem by optimizing over a series of subspace,
uncomment the two subspace statements below. Note that
the evaluations routines myvalue, mygrad, and myvalgrad currently
do not exploit the fact that x is only changing in a subspace */

#include "driver.h"

int BLG_main (int n, double *x, double *lo, double *hi, double *a, double b, \
             double norm, double *q, double *alpha_list, double C, int yi)
//int BLG_main (int n, double *x, double *lo, double *hi, double *a, double b)
{
    //double *x, *lo, *hi, *a, b ;
    //int i, n, n1, n2 ;
    int i;
    BLGparm Parm ;

    /* allocate space for solution */
    /*
    n = 100 ;
    */
    /*
    n = 10;
    x  = (double *) malloc (n*sizeof (double)) ;
    lo = (double *) malloc (n*sizeof (double)) ;
    hi = (double *) malloc (n*sizeof (double)) ;
    a  = (double *) malloc (n*sizeof (double)) ;
     */

    /* set up the constraints */
    /*
    n1 = n/3 ;
    n2 = 2*n/3 ;
    for (i = 0; i < n1; i++) { lo [i] = 0. ; hi [i] = 2. ; }
    for (; i < n2; i++) { lo [i] = 1. ; hi [i] = 3. ; }
    for (; i < n; i++) { lo [i] = 2. ; hi [i] = 4. ; }
    for (i = 0; i < n; i++) a [i] = 1. ;
    for (i = n1; i < n2; i++) a [i] = -1. ;
     */
    /*
    for(i = 0; i < n; ++i)
    {
        lo[i] = 0. + 1e-50;
        hi[i] = 1. - 1e-50;
        a[i] = 1.;
    }
     */

    /* set starting guess */
    /*
    for (i = 0; i < n; i++) x [i] = 2. ;
     */
    /*
    for (i = 0; i < n; i++) x [i] = (-1e-4)/9 ;
    x[0] = 1e-4;
     */

    /* for this test problem, we take b = a'x */
    /*
    b = 0. ;
    for (i = 0; i < n; i++) b += a [i]*x [i] ;
     */

    BLGdefault(&Parm) ;
    Parm.PrintFinal = BLGFALSE ;
/*  Parm.Subspace = BLGTRUE ;
    Parm.nsub = 10 ;*/

    /* run the code */
    BLG (x, NULL, lo, hi, a, b, NULL, n, 1.e-12, NULL, NULL, NULL, &Parm,
         NULL, myvalue, mygrad, myvalgrad, norm, q, alpha_list, C, yi) ;

    /* with some loss of efficiency, you could omit the valgrad routine */
    /*
    Parm.PrintParms = 0 ;   // do not print the parameters this time
    printf ("\nUSE affine scaling search direction\n") ;
    Parm.GP = 0 ;           // use affine scaling direction
    for (i = 0; i < n; i++) x [i] = 2. ; // starting guess
    BLG (x, NULL, lo, hi, a, b, NULL, n, 1.e-12, NULL, NULL, NULL, &Parm,
         NULL, myvalue, mygrad, NULL, norm, q, alpha_list, C) ;
    */

    /*
    free (x) ;
    free (a) ;
    free (lo) ;
    free (hi) ;
     */
    return (1) ;
}

void myvalue
(
    BLGobjective *user
)
{
    /*
    double f, t, *x ;
    int i, n ;
    x = user->x ;
    n = user->n ;
    f = 0. ;
    for (i = 0; i < n; i++)
    {
        t = i+1 ;
        t = sqrt (t) ;
        f += exp (x [i]) - t*x [i] ;
    }
    user->f = f ;
    return ;
    */

    double f = 0.0;
    double *x = user->x;
    int n = user->n;
    double value;
    double log_value;
    double *alpha_list = user->alpha_list;
    double *q = user->q;
    double norm = user->norm;
    double C = user->C;
    int yi = user->yi;

    for(int i = 0; i < n; ++i)
    {
        value = x[i];
        if(i == yi)
        {
            log_value = log(1-alpha_list[i]-value);
            f += (1-alpha_list[i]-value) * log_value;
        } else
        {
            log_value = log(-alpha_list[i]-value);
            f += (-alpha_list[i]-value) * log_value;
        }

        f += 0.5 * C * value * value * norm;
        f += C * q[i] * value;
    }
    user->f = f;

    return;
}

void mygrad
(
    BLGobjective *user
)
{
    /*
    double t, *g, *x ;
    int i, n ;
    x = user->x ;
    g = user->g ;
    n = user->n ;
    for (i = 0; i < n; i++)
    {
        t = i + 1 ;
        t = sqrt (t) ;
        g [i] = exp (x [i]) -  t ;
    }
    return ;
    */

    double *x = user->x;
    double *g = user->g;
    int n = user->n;
    double value;
    double log_value;

    double *alpha_list = user->alpha_list;
    double *q = user->q;
    double norm = user->norm;
    double C = user->C;
    int yi = user->yi;

    for(int i = 0; i < n; ++i)
    {
        value = x[i];
        if(i == yi)
        {
            log_value = log(1-alpha_list[i]-value);
        } else
        {
            log_value = log(-alpha_list[i]-value);
        }

        g[i] = C*norm*value + C*q[i] + 1 - log_value;
    }

    return;
}

void myvalgrad
(
    BLGobjective *user
)
{
    /*
    double ex, f, t, *g, *x ;
    int i, n ;
    f = (double) 0 ;
    x = user->x ;
    g = user->g ;
    n = user->n ;
    for (i = 0; i < n; i++)
    {
        t = i + 1 ;
        t = sqrt (t) ;
        ex = exp (x [i]) ;
        f += ex - t*x [i] ;
        g [i] = ex -  t ;
    }
    user->f = f ;
    return ;
     */
    double f = 0.0;
    double *x = user->x;
    double *g = user->g;
    int n = user->n;
    double value;
    double log_value;

    double *alpha_list = user->alpha_list;
    double *q = user->q;
    double norm = user->norm;
    double C = user->C;
    int yi = user->yi;

    for(int i = 0; i < n; ++i)
    {
        value = x[i];
        if(i == yi)
        {
            log_value = log(1-alpha_list[i]-value);
            f += (1-alpha_list[i]-value) * log_value;
        } else
        {
            log_value = log(-alpha_list[i]-value);
            f += (-alpha_list[i]-value) * log_value;
        }

        f += 0.5 * C * value * value * norm;
        f += C * q[i] * value;

        g[i] = C*norm*value + C*q[i] + 1 - log_value;
    }
    user->f = f;

    return;
}
