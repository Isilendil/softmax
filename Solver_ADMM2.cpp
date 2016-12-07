//////////////////////////////////////////////
//dual coordinate descent for softmax regression
#include "Solver_ADMM2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <algorithm>
#include <time.h>

#include <iostream>
//////////////////////////////////////////////

Solver_ADMM2::Solver_ADMM2(const problem *prob, int nr_class, double C, double eps, int max_iter, \
  double initial, int max_inner_iter, int max_newton_iter)
{
	this->prob = prob;
	this->nr_class = nr_class;
	this->C = C;
	this->eps = eps;
	this->max_iter = max_iter;


	this->w_size = prob->n * nr_class;
	this->l = prob->l;
	this->n = prob->n;

    this->pi = initial;

    alpha_size = l * nr_class;
	alpha = new double[alpha_size];


	this->max_inner_iter = max_inner_iter * nr_class;
    this->max_newton_iter = max_newton_iter;


	norm_list = new double[l];
	for(int i = 0; i < l; i++)
	{
		norm_list[i] = 0;
	}
}

Solver_ADMM2::~Solver_ADMM2()
{
	delete [] alpha;
	//delete [] alpha_new;
	delete [] norm_list;
}
void Solver_ADMM2::Solve(double *w, double *obj, Function_SOFTMAX *func)
{
	/*
	int *perm = Malloc(int, l);
    double *obj = Malloc(double, max_iter+1);
	double *grad = Malloc(double, w_size);
	double *probability = Malloc(double, nr_class);
	double *norm_term = Malloc(double, nr_class);
	*/
	int *perm = new int[l];
    double *grad_w = new double[w_size];
	//double *obj = new double[max_iter+1];
	//double *grad = new double[nr_class];
	//double *output = new double[nr_class];
	//double *probability = new double[nr_class];
	//double *norm_term = new double[nr_class];
    //double sum = 0;
    //int id;
	//double *v = new double[nr_class];

    // variables for sub-problem
	double *d = new double[nr_class];
    double *z = new double[nr_class];
    double *lambda = new double[nr_class+1];
    double rho = 1;

    /*
    // gradient descent
    double rho = 1e-6;
    */
    double eps_z = 1e-8;
    double eps_lambda = 1e-8;

    double *q = new double[nr_class];
    double *G = new double[nr_class];
    double *GG = new double[nr_class];


    double timer = 0;
    double grad_norm;
    double primal;
    double dual;
    double accuracy;

    double factor = 0.99;

    double *alpha_new = new double [alpha_size];
    //initialize alpha
    for(int i = 0; i < alpha_size; i++)
	{
		alpha[i] = pi / (1-nr_class);
	}
	for(int id = 0; id < l; id++)
	{
		int yi = prob->y[id];
		alpha[id*nr_class+yi] = pi;
	}

    //initialize w
	//pre-calculate norm_list
	for(int i = 0; i < w_size; i++)
		w[i] = 0;
	
	for(int id = 0; id < l; id++)
	{
		feature_node *xi = prob->x[id];
		int yi = prob->y[id];

		while(xi->index != -1)
		{
			int index = xi->index - 1;
			double value = xi->value;

            //calculate norm_list
			norm_list[id] += value * value;

			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				w[index*nr_class+iter_class] += alpha[id*nr_class+iter_class] * value;
			}
			++xi;
		}
	}
		
    //compute objective
	obj[0] = compute_obj(w);



	int iter = 0;
	for(; iter < max_iter; iter++)
	//for(; iter < 1; iter++)
	{
        double start = clock();

        //rho *= 0.9999;
        //rho = std::max(rho, 1e-6 * 0.995);

		//permutation
		for(int i = 0; i < l; i++)
		{
			perm[i] = i;
		}
		for(int i = 0; i < l; i++)
		{
			int j = i + rand()%(l-i);
			//swap(perm[i], perm[j]);
			int temp = perm[i];
			perm[i] = perm[j];
			perm[j] = temp;
		}
		

		//two-level dual coordinate descent
		//the outer level considers a block of variables corresponding an instance , i.e., alpha_i
		for(int order = 0; order < l; order++)
		{
			int id = perm[order];
			feature_node *xi;
			int yi = prob->y[id];

            double norm_id = norm_list[id];

            //solve the sub-problem
            //initialize d, z, and lambda
            for(int i = 0; i < nr_class; i++)
            {
                d[i] = 0;
                z[i] = 0;
                lambda[i] = 0;
            }
            lambda[nr_class] = 0;

            //compute q
			for(int i = 0; i < nr_class; i++)
			{
				q[i] = 0;
			}
			xi = prob->x[id];
			while(xi->index != -1)
			{
				int index = xi->index - 1;
				double value = xi->value;
				for(int i = 0; i < nr_class; i++)
				{
					q[i] += w[index*nr_class+i] * value;
				}
				++xi;
			}

            //inner loop for sub-problem
            for(int inner_iter = 0; inner_iter < max_inner_iter; inner_iter++)
            {
                //update d
                for(int newton_iter = 0; newton_iter < max_newton_iter; newton_iter++)
                {
                    //compute first-order derivatives
                    double sum_d = 0;
                    double alpha_temp;
                    for(int i = 0; i < nr_class; i++)
                    {
                        sum_d += d[i];
                    }
                    for(int i = 0; i < nr_class; i++)
                    {
                        //compute first-order derivatives
                        alpha_temp = alpha[id*nr_class+i];
                        G[i] = C*norm_list[id]*d[i] + rho*(sum_d+d[i]);
                        G[i] += C*q[i] + lambda[0]+lambda[i] - rho*z[i];

                        //compute second-order derivatives
                        //diagonal approximation
                        GG[i] = C*norm_list[id] + rho*2;
                    }


                    //inverse Hessian matrix
                    //

                    //update
                    for(int i = 0; i < nr_class; i++)
                    {
                        //d[i] -= rho * G[i]/GG[i];
                        //d[i] -= G[i]/GG[i];
                        //d[i] -= rho * G[i];
                        double d_temp = d[i] - 0.1 * G[i]/GG[i];
                        //double d_temp = d[i] - 0.000000001 * G[i];
                        d[i] = d_temp;


                    }
                }

                //update z
                double z_temp;
                double z_gap = 0;
                double alpha_temp;
                for(int i = 0; i < nr_class; i++)
                {
                    G[i] = lambda[i+1] + rho*z[i] - rho*d[i];

                    alpha_temp = alpha[id*nr_class+i];
                    if(i == yi)
                    {
                        alpha_temp = 1 - alpha_temp - z[i];
                    }
                    else
                    {
                       alpha_temp = -alpha_temp - z[i];
                    }
                    G[i] += 1 - log(alpha_temp);

                    GG[i] = rho + 1/alpha_temp;

                    z_temp = z[i] - 0.1 * G[i]/GG[i];
                    //z_temp = z[i] - 0.00000001 * G[i];

                    //projection
                    //alpha_temp = alpha[id*nr_class+i] + z_temp;
                    double alpha_old;
                    alpha_old = alpha[id*nr_class+i];
                    alpha_temp = alpha_old + z_temp;
                    if (i == yi)
                    {
                        if (alpha_temp <= 0)
                        {
                            alpha_temp = alpha_old * (1-factor);
                        }
                        else if (alpha_temp >= 1)
                        {
                            alpha_temp = alpha_old*(1-factor) + factor;
                        }
                        //log_alpha = (1-alpha_temp) * log(1-alpha_temp);
                    }
                    z_temp = alpha_temp - alpha[id*nr_class+i];
                    z_gap += (z[i]-z_temp) * (z[i]-z_temp);
                    z[i] = z_temp;
                }

                //update lambda
                double sum_d = 0;
                double lambda_temp;
                double lambda_gap = 0;
                for(int i = 1; i < nr_class+1; i++)
                {
                    sum_d += d[i-1];

                    lambda_temp = rho * (d[i-1]-z[i-1]);
                    lambda_gap += lambda_temp * lambda_temp;
                    lambda[i] += lambda_temp;
                }
                lambda_temp = rho * sum_d;
                lambda_gap += lambda_temp;
                lambda[0] += lambda_temp;

                if( (z_gap <= eps_z) && (lambda_gap <= eps_lambda))
                    break;

            }
            //end of inner loop

            //update w
            xi = prob->x[id];
            while(xi->index != -1)
            {
                int index = xi->index - 1;
                double value = xi->value;
                double temp = 0;
                for(int iter_class = 0; iter_class < nr_class; iter_class++)
                {
                    w[index*nr_class+iter_class] += z[iter_class] * value;
                }
                ++xi;
            }
            //update alpha
            for(int iter_class = 0; iter_class < nr_class; iter_class++)
            {
                alpha[id*nr_class+iter_class] += z[iter_class];
            }

        }
		// one epoch of dual coordinate descent

        timer += clock() - start;

        primal = func->obj_primal(w);


        for(int i = 0; i < alpha_size; i++)
        {
            alpha_new[i] = -alpha[i];
        }
        for(int id = 0; id < l; id++)
        {
            int yi = prob->y[id];
            alpha_new[id*nr_class+yi] = alpha[id*nr_class+yi] + 1;
        }
        dual = func->obj_dual(alpha_new, w);

        func->grad(w, grad_w, &grad_norm);

        accuracy = func->testing(w);

        std::cout << iter << '\t' << timer << '\t' << primal << '\t' << dual << '\t' << accuracy << '\t' << grad_norm << std::endl;

        //compute objective
        obj[iter+1] = compute_obj(w);

        /*
        if(iter>5 && (fabs(obj[iter+1]-obj[iter])/obj[iter] <= eps) )
        {
            break;
        }
         */
	}

	obj[iter+1] = -1;

	//delete [] norm_term;
	//delete [] probability;
	//delete [] grad;
	//delete [] obj;
    delete [] perm;
    delete [] grad_w;
	//delete [] output;
	delete [] G;
    delete [] GG;

	delete [] d;
    delete [] z;
    delete [] lambda;
    delete [] q;

    delete [] alpha_new;

}



double Solver_ADMM2::compute_obj(double *w)
{
	double obj = 0;

	// ||W||_F
	for(int i = 0; i < w_size; i++)
	{

		obj += w[i] * w[i];
	}
	obj *= 0.5;

  // loss
  double loss_instance = 0;
	//double *output = Malloc(double, nr_class);
	double *output = new double[nr_class];
  double sum = 0;
  double normal_term = 0;
 
  for(int id = 0; id < l; id++)
	{
		for(int i = 0; i < nr_class; i++)
		{
			output[i] = 0;
		}

		feature_node *xi = prob->x[id];
		while(xi->index != -1)
		{
			int index = xi->index-1;
			double value = xi->value;
			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				output[iter_class] += w[index*nr_class+iter_class] * value;
				//output[iter_class] += w[iter_class*n+index] * value;
			}
      ++xi;
		}
 
    sum = 0;
		for(int i = 0; i < nr_class; i++)
		{
			sum += exp(output[i]);
		}
		sum = log(sum);
    
		int yi = prob->y[id];
		loss_instance -= output[yi]-sum;

	}

	loss_instance *= C;

	obj += loss_instance;

  delete [] output;

	return obj;
}


/*
double Solver_ME_DUAL::compute_dual_dif(double *w)
{
	double dual_dif = 0;
  
	double term_1 = 0;
	double term_2 = 0;
	double term_3 = 0;
	double term_4 = 0;

  double alpha_dif = 0;

	feature_node *xi = prob->x[id_current];
	int yi = prob->y[id_current];
	double norm_xi = 0;
	while(xi->index != -1)
	{
		norm_xi += xi->value * xi->value;
		++xi;
	}

	for(int iter_class = 0; iter_class < nr_class; iter_class++)
	{
		alpha_dif = alpha_new[iter_class] - alpha[id_current*nr_class+iter_class];

		term_1 = alpha_new[iter_class] * log(alpha_new[iter_class]);
		term_2 = alpha[id_current*nr_class+iter_class] * log(alpha[id_current*nr_class+iter_class]);
		
		term_3 = 0;
		xi = prob->x[id_current];
		while(xi->index != -1)
		{
			int index = xi->index - 1;
			double value = xi->value;
			term_3 += w[index*nr_class+iter_class] * value;
			++xi;
		}
		term_3 *= alpha_dif;

    term_4 = 0.5 * alpha_dif*alpha_dif * norm_xi;

    dual_dif += term_1 - term_2 + term_3 + term_4;
	}

	return dual_dif;
}
*/

//////////////////////////////////////////////
