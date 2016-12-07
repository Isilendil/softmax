///////////////////////////////////////////////////////
//softmax

#ifndef LIBLINEAR_TEST_SOLVER_SGD_H
#define LIBLINEAR_TEST_SOLVER_SGD_H

#include "linear.h"
#include "Function_SOFTMAX.h"


class Solver_SGD
{
	public:
	  Solver_SGD(const problem *prob, int nr_class, double C, double eta, \
	    double eps = 0.00001, int max_iter = 1000);
		//eta = 0.000001 for pendigits
		//eta = 0.01 for real-sim will generate NaN error
		~Solver_SGD();
		void Solve(double *w, double *obj, Function_SOFTMAX *func);

	int max_iter;

	private:
	void solve_sub_problem();
	double compute_obj(double *w);
	int w_size;
	int l;
	int nr_class;
	double eps;
	double C;
	double eta;

	const problem *prob;
	int n;

	// armijo rule
	double delta;
	double beta;
	bool armijo;
    int max_armijo;
};
#endif //LIBLINEAR_TEST_SOLVER_SGD_H

///////////////////////////////////////////////////////
