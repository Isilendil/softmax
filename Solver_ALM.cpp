//////////////////////////////////////////////
//dual coordinate descent for softmax regression
#include "Solver_ALM.h"
#include <math.h>
//#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
//#include <stdarg.h>
//#include <locale.h>
#include <algorithm>
//#include <time.h>

#include <iostream>
//////////////////////////////////////////////
bool debug_flag = false;

Solver_ALM::Solver_ALM(const problem *prob, int nr_class, double C, double eps, int max_iter, \
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

Solver_ALM::~Solver_ALM()
{
	delete [] alpha;
	//delete [] alpha_new;
	delete [] norm_list;
}
void Solver_ALM::Solve(double *w, double *obj, Function_SOFTMAX *func)
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
    double rho;
    double rho_max = 5;
    double rho_init = 1;
    //double rho = 1e-1;
    double eps_lambda = 1e-2;
    double eps_G = 1e-2 * 2;
    double factor = 0.9;

    //bool line_search = false;
    bool newton;
    double beta = 1.00;
    double eta = 0.2;
    double eta_g_upper = 1e-4;
    double eta_g_lower = 1e-6;
    double G_norm;
    double G_norm_old;
    double G_gap;
    double sigma = 1e-1;
    double sub_obj;
    double sub_obj_new;
    double log_alpha;
    double term1;
    double term2;
    double *d_temp = new double[nr_class];

    double *sk = new double[nr_class];
    double *yk = new double[nr_class];

    double G_temp;
    double yk_sum = 0;
    double sk_sum = 0;
    double *G_log_dif = new double[l];
    //double *eta_list = new double[l];
    //for(int i = 0; i < l; ++i)
    //{
        //eta_list[i] = eta;
    //}
    /*
    // gradient descent
    double rho = 1e-6;
    */

    double *q = new double[nr_class];
    double *G = new double[nr_class];
    double *GG = new double[nr_class];


    double timer = 0;
    double grad_norm;
    double primal;
    double dual;
    double accuracy;


    double *alpha_new = new double [alpha_size];
    double *G_log = new double [alpha_size];

    int i;
    int iter_class;
    int id;
    feature_node *xi;
    int yi;

    int iter = 0;
    int inner_iter;
    int newton_iter;
    int j;
    int temp;

    int index;
    double value;
    double start;

    int order;
    double norm_id;
    double lambda;

    double alpha_temp;
    double alpha_old;
    double eta_temp;

    double sum_d = 0;
    double lambda_gap = 0;
    //initialize alpha
    for(i = 0; i < alpha_size; i++)
	{
		alpha[i] = pi / (1-nr_class);
	}
	for(id = 0; id < l; id++)
	{
		yi = prob->y[id];
		alpha[id*nr_class+yi] = pi;
	}

    //initialize w
	//pre-calculate norm_list
	for(i = 0; i < w_size; i++)
		w[i] = 0;
	
	for(id = 0; id < l; id++)
	{
		xi = prob->x[id];
		yi = prob->y[id];

		while(xi->index != -1)
		{
			index = xi->index - 1;
			value = xi->value;

            //calculate norm_list
			norm_list[id] += value * value;

			for(iter_class = 0; iter_class < nr_class; iter_class++)
			{
				w[index*nr_class+iter_class] += alpha[id*nr_class+iter_class] * value;
			}
			++xi;
		}
	}
		
    //compute objective
	obj[0] = compute_obj(w);



	for(; iter < max_iter; iter++)
	//for(; iter < 1; iter++)
	{
        start = clock();

        //rho *= 0.9999;
        beta = std::max(0.3, beta * 0.90);

		//permutation
		for(i = 0; i < l; ++i)
		{
			perm[i] = i;
		}
		for(i = 0; i < l; ++i)
		{
			j = i + rand()%(l-i);
			//swap(perm[i], perm[j]);
			temp = perm[i];
			perm[i] = perm[j];
			perm[j] = temp;
		}


		//two-level dual block-coordinate descent
		//the outer level considers a block of variables corresponding an instance , i.e., alpha_i
		for(order = 0; order < l; ++order)
		{
            rho = rho_init;

			id = perm[order];
			yi = prob->y[id];

            norm_id = norm_list[id];

            //solve the sub-problem
            //initialize d and lambda
            //compute q
            lambda = 0.0;
            for(i = 0; i < nr_class; ++i)
            {
                d[i] = 0;
                q[i] = 0;
                G[i] = 0;
            }
            newton = true;

			xi = prob->x[id];
			while(xi->index != -1)
			{
				index = xi->index - 1;
				value = xi->value;
				for(i = 0; i < nr_class; ++i)
				{
					q[i] += w[index*nr_class+i] * value;
				}
				++xi;
			}

            //inner loop for sub-problem
            for(inner_iter = 0; inner_iter < max_inner_iter; ++inner_iter, rho *= 1.1)
            {
                rho = std::min(rho, rho_max);
                //update d
                for(newton_iter = 0; newton_iter < max_newton_iter; ++newton_iter)
                {

                    if(newton_iter == 1)
                    {
                        newton = false;
                    }
                    if(newton_iter == 3)
                    {
                        newton = true;
                    }

                    //compute derivatives
                    sum_d = 0;
                    for(i = 0; i < nr_class; ++i)
                    {
                        sum_d += d[i];
                    }
                    G_norm = 0;
                    //sub_obj = 0;
                    //term1 = 0;
                    //term2 = 0;
                    yk_sum = 0;
                    for(i = 0; i < nr_class; ++i)
                    {
                        //compute first-order derivatives
                        alpha_temp = alpha[id*nr_class+i];
                        G_temp = C*norm_id*d[i] + rho*sum_d;
                        G_temp += C*q[i] + lambda;


                        //if(inner_iter == 0 && newton_iter == 0)
                        if (i == yi)
                        {
                            alpha_temp = 1 - alpha_temp - d[i];
                        } else
                        {
                            alpha_temp = -alpha_temp - d[i];
                        }
                            //term1 += d[i] * d[i];
                            //term2 += (C*q[i]+lambda) * d[i];

                        log_alpha = log(alpha_temp);
                            //sub_obj += alpha_temp * log_alpha;

                        G_temp += 1 - log_alpha;

                        //compute second-order derivatives
                        //diagonal approximation

                        G_norm += G_temp * G_temp;

                        yk[i] = G_temp - G[i];
                        G[i] = G_temp;

                        yk_sum += fabs(yk[i] * sk[i]);

                        if(newton)
                        {
                            GG[i] = C * norm_list[id] + rho * 1 + 1 / alpha_temp;
                        }
                    }
                    G_gap = fabs(G_norm-G_norm_old) / G_norm_old;
                    if(G_gap <= eps_G)
                    {
                       break;
                    }
                    else
                    {
                        G_norm_old = G_norm;
                    }
                    //sub_obj += 0.5 * term1 * (C*norm_id*norm_id+rho) + term2;

                    //update d
                    //sub_obj_new = 0;
                    //term1 = 0;
                    //term2 = 0;
                    for(i = 0; i < nr_class; ++i)
                    {
                        //d[i] -= rho * G[i]/GG[i];
                        //d[i] -= G[i]/GG[i];
                        //d[i] -= rho * G[i];
                        //double d_temp = d[i] - 1.0* G[i]/GG[i];
                        //beta = 0.5;

                        // project
                        //d_temp[i] = d[i] - eta * G[i];
                        if(newton)
                        {
                            d_temp[i] = d[i] - beta * eta * G[i] / GG[i];
                        } else
                        {
                            eta_temp = sk_sum / yk_sum;
                            eta_temp = std::max(eta_temp, eta_g_lower);
                            eta_temp = std::min(eta_temp, eta_g_upper);
                            //eta_temp = eta_g;
                            d_temp[i] = d[i] - beta * G[i] * eta_temp;
                        }
                        alpha_old = alpha[id*nr_class+i];
                        alpha_temp = alpha_old + d_temp[i];

                        factor = newton_iter / (10.0+newton_iter);
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
                        else
                        {
                            if (alpha_temp >= 0)
                            {
                                alpha_temp = alpha_old * (1-factor);
                            }
                            else if (alpha_temp <= -1)
                            {
                                alpha_temp = alpha_old * (1-factor) - factor;
                            }
                            //log_alpha = (-alpha_temp) * log(-alpha_temp);
                        }
                        //alpha_new[i] = alpha_temp;
                        d_temp[i] = alpha_temp - alpha_old;
                        //term1 += d_temp[i] * d_temp[i];
                        //term2 += (C*q[i]+lambda) * d_temp[i];
                        //sub_obj_new += log_alpha;
                    }
                    //sub_obj_new += 0.5 * term1 * (C*norm_id*norm_id+rho) + term2;

                    // line search


                    sk_sum = 0;
                    for(i = 0; i < nr_class; ++i)
                    {
                        sk[i] = d_temp[i] - d[i];
                        d[i] = d_temp[i];
                        //sk_sum += sk[i]*sk[i]/GG[i];
                        sk_sum += sk[i]*sk[i];
                        //d[i] = alpha_new[i] - alpha[id*nr_class+i];
                    }
                }

                if(debug_flag)
                std::cout << newton_iter << '\t' << G_gap << '\t' << lambda_gap << '\t' << clock()-start <<  std::endl;

                //update lambda
                sum_d = 0;
                for(i = 0; i < nr_class; ++i)
                {
                    sum_d += d[i];
                }
                lambda_gap = rho * sum_d;
                lambda += lambda_gap;

                if( fabs(lambda_gap) <= eps_lambda)
                    break;

            }
            //end of inner loop
            if(debug_flag)
            std::cout << inner_iter << '\t'  << G_gap << '\t' << lambda_gap << '\t' << clock()-start << std::endl;

            //update w
            xi = prob->x[id];
            while(xi->index != -1)
            {
                index = xi->index - 1;
                value = xi->value;
                for(iter_class = 0; iter_class < nr_class; ++iter_class)
                {
                    w[index*nr_class+iter_class] += d[iter_class] * value;
                }
                ++xi;
            }
            //update alpha
            for(iter_class = 0; iter_class < nr_class; ++iter_class)
            {
                alpha[id*nr_class+iter_class] += d[iter_class];
            }

        }
		// one epoch of dual coordinate descent

        timer += clock() - start;

        primal = func->obj_primal(w);


        for(i = 0; i < alpha_size; ++i)
        {
            alpha_new[i] = -alpha[i];
        }
        for(id = 0; id < l; ++id)
        {
            yi = prob->y[id];
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
    delete [] q;

    delete [] alpha_new;
    delete [] G_log;
    delete [] d_temp;
    delete [] G_log_dif;

    delete [] sk;
    delete [] yk;

}



double Solver_ALM::compute_obj(double *w)
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
