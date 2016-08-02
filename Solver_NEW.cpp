//////////////////////////////////////////////
//dual coordinate descent for softmax regression
#include "Solver_NEW.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <algorithm>
//////////////////////////////////////////////

Solver_NEW::Solver_NEW(const problem *prob, int nr_class, double C, double eps, int max_iter, double eta_value, int max_trial, double initial, int max_inner_iter, int update_index_size)
{
	this->prob = prob;
	this->nr_class = nr_class;
	this->C = C;
	this->eps = eps;
	this->max_iter = max_iter;


	this->w_size = prob->n * nr_class;
	this->l = prob->l;
	this->n = prob->n;

  alpha_size = l * nr_class;
	alpha = new double[alpha_size];
	
	//alpha_new = new double[nr_class];

  this->max_trial = max_trial;
	this->initial = initial;


	this->max_inner_iter = max_inner_iter;

  this->update_index_size = update_index_size;

	norm_list = new double[l];
	for(int i = 0; i < l; i++)
	{
		norm_list[i] = 0;
	}
}

Solver_NEW::~Solver_NEW()
{
	delete [] alpha;
	//delete [] alpha_new;
	delete [] norm_list;
}

void Solver_NEW::Solve(double *w, double *obj)
{
	/*
	int *perm = Malloc(int, l);
  double *obj = Malloc(double, max_iter+1);
	double *grad = Malloc(double, w_size);
	double *probability = Malloc(double, nr_class);
	double *norm_term = Malloc(double, nr_class);
	*/
	int *perm = new int[l];
	//double *obj = new double[max_iter+1];
	double *grad = new double[nr_class];
	double *output = new double[nr_class];
	//double *probability = new double[nr_class];
	//double *norm_term = new double[nr_class];
  //double sum = 0;
  //int id;
	double *v = new double[nr_class];
	double *z = new double[nr_class];
	double *G = new double[nr_class];
	double *GG = new double[nr_class];

  // variables for sub-problem
	double *d = new double[update_index_size];
	double *upper = new double[update_index_size];
	double *lower = new double[update_index_size];
  int *update_index = new int[update_index_size];

  double g_eps = 0.5;

  double dif;
  int inner_iter;

  //initialize alpha
  for(int i = 0; i < alpha_size; i++)
	{
		alpha[i] = initial / (nr_class-1);
	}
	for(int id = 0; id < l; id++)
	{
		int yi = prob->y[id];
		alpha[id*nr_class+yi] = 1 - initial;
	}

  //initialize w
	//Eq. (41)
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
				if(iter_class != yi)
				{
				  w[index*nr_class+iter_class] += (-alpha[id*nr_class+iter_class] * value);
				}
			}
			w[index*nr_class+yi] += ((1-alpha[id*nr_class+yi]) * value);
			++xi;
		}
	}
		
  //compute objective
	obj[0] = compute_obj(w);


	int iter = 0;
	for(; iter < max_iter; iter++)
	//for(; iter < 1; iter++)
	{
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
			id_current = id;

      double norm_id = norm_list[id];

      //solve the sub-problem (42) by Algorithm 6 and get the optimal z

			//compute v
			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				v[iter_class] = 0;
			}
			xi = prob->x[id];
			while(xi->index != -1)
			{
				int index = xi->index - 1;
				double value = xi->value;
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
					v[iter_class] += w[index*nr_class+iter_class] * value;
				}
				//compute K^i = norm(xi)^2
				//norm_list[id] += value * value;

				++xi;
			}

			//initialize z
			double z_max = -INF;
			int z_max_iter;

			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				z[iter_class] = alpha[id*nr_class+iter_class];
				if(z[iter_class] > z_max)
				{
					z_max = z[iter_class];
					z_max_iter = iter_class;
				}
			}
			
			//compute initial gradient
			//record G_max and G_min
			double G_max = -INF;
			int G_max_iter = -1;
			double G_min = INF;
			int G_min_iter = -1;

			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				G[iter_class] = log(z[iter_class]) + 1 - v[iter_class];

				if(iter_class == z_max_iter)
					continue;

				if(G[iter_class] < G_min)
				{
					G_min = G[iter_class];
					G_min_iter = iter_class;
					//continue;
				}
			}

			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				//first-order 
				if(G[iter_class] > G_max)
				{
					G_max = G[iter_class];
					G_max_iter = iter_class;
					//continue;
				}
      }

			//compute second-order derivatives
			//record GG_max
			double GG_max = -INF;
			int GG_max_iter = -1;
			double GG_min = INF;
			int GG_min_iter = -1;

			for(int iter_class_rev = nr_class-1; iter_class_rev >= 0; iter_class_rev--)
			{
				//second-order
				if(iter_class_rev == z_max_iter)
					continue;

				GG[iter_class_rev] = 1/(z[iter_class_rev]);
				if(iter_class_rev == G_max_iter || iter_class_rev == G_min_iter)
					continue;

				if(GG[iter_class_rev] > GG_max)
				{
					GG_max = GG[iter_class_rev];
					GG_max_iter = iter_class_rev;
					//continue;
				}
			}
			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				if(iter_class == z_max_iter)
					continue;

				if(iter_class == G_max_iter || iter_class == G_min_iter || iter_class == GG_max_iter)
					continue;

				if(GG[iter_class] < GG_min)
				{
					GG_min = GG[iter_class];
					GG_min_iter = iter_class;
					//continue;
				}
			}


      //the inner loop solves the sub-problem
			for(inner_iter = 0; inner_iter < max_inner_iter; inner_iter++)
			//for(int inner_iter = 0; inner_iter < nr_class; inner_iter++)
			{
				//if(G_max == G_min)
				//dif = (G_max-G_min) / std::min(fabs(G_min),fabs(G_max));
				dif = fabs((G_max-G_min)/G_min);
				if( dif < g_eps )
				{
					break;
				}

			  //construct update_index
        update_index[0] = G_min_iter;
			  //update_index[3] = G_max_iter;
			  update_index[2] = GG_max_iter;
        update_index[1] = GG_min_iter;
				update_index[3] = z_max_iter;

        //calculate coefficients 
				double a;
				double b;
				double c;
        for(int update_iter = 0; update_iter < update_index_size; update_iter++)
				{
					int temp = update_index[update_iter];
					a = z[temp];
					b = norm_id;
					c = z[temp] - alpha[id*nr_class+update_index[update_iter]];
          c = c * norm_id - v[update_index[update_iter]];

          upper[update_iter] = 1 - a;
					lower[update_iter] = -a;

					solve_sub_problem(a, b, c, &(d[update_iter]));
				}

				//adjust d to satisfy equality constraint
				adjust_stepsize(d, upper, lower);

        
				 
				//update z
				//update first-order derivatives
				//update second-order derivatives
				for(int update_iter = 0; update_iter < update_index_size; update_iter++)
				{
					int index_temp = update_index[update_iter];

					double temp = z[index_temp] + d[update_iter];
					z[index_temp] = temp;

					//z[index_temp] = double(0.1234);
					//z[index_temp] = z[index_temp] + d[update_iter];
					//z[index_temp] += d[update_iter];
					//double temp_2 = 0.123;
					//z[index_temp] = temp*0.1*10;
					//z[index_temp] *= 10;
					//d[update_iter] = 0.1;

				  G[index_temp] = log(z[index_temp]) + 1 + norm_id * (z[index_temp]-alpha[id*nr_class+index_temp])- v[index_temp];

				  GG[index_temp] = 1/(z[index_temp]) + norm_id;
				}
        
				
				z_max = - INF;
				z_max_iter = -1;
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
				  if(z[iter_class] > z_max)
				  {
					  z_max = z[iter_class];
					  z_max_iter = iter_class;
				  }
				}

				//find G_max and G_min
				G_max = -INF;
				G_min = INF;
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
				  if(iter_class == z_max_iter)
					  continue;

				  if(G[iter_class] < G_min)
				  {
					  G_min = G[iter_class];
					  G_min_iter = iter_class;
				  }
				}

				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
				  if(G[iter_class] > G_max)
				  {
					  G_max = G[iter_class];
					  G_max_iter = iter_class;
				  }
				}
	      //find GG_max
	      for(int iter_class_rev = nr_class-1; iter_class_rev >= 0; iter_class_rev--)
			  {
				if(iter_class_rev == z_max_iter)
					continue;

				  if(iter_class_rev == G_max_iter || iter_class_rev == G_min_iter)
					  continue;

				  if(GG[iter_class_rev] > GG_max)
				  {
					  GG_max = GG[iter_class_rev];
					  GG_max_iter = iter_class_rev;
						//continue;
				  }
				}
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
				if(iter_class == z_max_iter)
					continue;

				  if(iter_class == G_max_iter || iter_class == G_min_iter || iter_class == GG_max_iter)
					  continue;

				  if(GG[iter_class] < GG_min)
				  {
					  GG_min = GG[iter_class];
					  GG_min_iter = iter_class;
						//continue;
				  }
			  }


			}
			//end of solving the sub-problem

			//update w
			xi = prob->x[id];
			while(xi->index != -1)
			{
				int index = xi->index - 1;
				double value = xi->value;
				double temp = 0;
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
					temp = (z[iter_class]-alpha[id*nr_class+iter_class]);
           w[index*nr_class+iter_class] -= temp * value;
				}
				++xi;
			}
			//update alpha 
      for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				alpha[id*nr_class+iter_class] = z[iter_class];
			}

		}
		// one epoch of dual coordinate descent
		
    //compute objective
		obj[iter+1] = compute_obj(w);

		if(iter>5 && (fabs(obj[iter+1]-obj[iter])/obj[iter] <= eps) )
		{
			break;
		}
	}

	obj[iter+1] = -1;

	//delete [] norm_term;
	//delete [] probability;
	delete [] grad;
	//delete [] obj;
  delete [] perm;
	delete [] output;
	delete [] v;
	delete [] z;
	delete [] G;
  delete [] GG;

	delete [] update_index;
	delete [] d;
	delete [] upper;
	delete [] lower;

}

// g(d) = (a+d)log(a+d) + 0.5*b*d^2 + c*d
void Solver_NEW::solve_sub_problem(double a, double b, double c, double *pd)
{
  double l = -a;
	double u = 1-a;

  double g_1;
	double g_2;

  double d = 0.0;
  double dd;

  const double eta = 0.1;
	const double innereps = 1e-5;

	// a modified Newton method
	int inner_iter;
  for(inner_iter = 0; inner_iter < max_inner_iter; inner_iter++)
	{
		// g'(d) = log(a+d) + 1 + b*d + c
		g_1 = log(a+d) + 1 + b*d + c;

		//if(fabs(g_1) < innereps)
		if(fabs(g_1) == 0)
		{
			break;
		}

		// g''(d) = 1/(a+d) + b
		g_2 = 1/(a+d) + b;

		dd = - g_1 / g_2;

		// check inequality constraint
		if(d+dd <= l)
		{
			d = eta*d + (1-eta)*l;
		}
		else if(d+dd >= u)
		{
			d = eta*d + (1-eta)*u;
		}
		else
		{
			d += dd;
		}
	}
	// end of Newton method

	*pd = d;

}

double Solver_NEW::compute_obj(double *w)
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

void Solver_NEW::adjust_stepsize(double *d, double *upper, double *lower)
{
	double d_pos = 0.0;
	double d_neg = 0.0;
  double d_dif;
  double d_dif_real;

  double eta = 0.9;

	// add up d_pos and d_neg
	for(int update_iter = 0; update_iter < update_index_size-1; update_iter++)
	{
		if(d[update_iter] > 0)
		{
			d_pos += d[update_iter];
		}
		else if(d[update_iter] < 0)
		{
			d_neg += -d[update_iter];
		}
	}

	// adjust d
  if(d_pos > d_neg)
	{
		d_dif = d_pos - d_neg;

		if(-d_dif <= lower[update_index_size-1])
		{
			//d_dif = d[update_index_size-1] - (0.1*d[update_index_size-1] + 0.9*low[update_index_size-1]);
			d_dif_real = eta * (0 - lower[update_index_size-1]);
		}
		else
		{
			d_dif_real = d_dif;
		}
		// adjust d_pos
		for(int update_iter = 0; update_iter < update_index_size-1; update_iter++)
		{
			if(d[update_iter] > 0)
			{
				d[update_iter] -= d[update_iter] * ((d_dif-d_dif_real)/d_pos);
			}
		}
	  d[update_index_size-1] = -d_dif_real;
	}

	else if(d_pos < d_neg)
	{
		d_dif = d_pos - d_neg;

		if(d_dif >= upper[update_index_size-1])
		{
			d_dif_real = eta * (upper[update_index_size-1] - 0);
		}
		else
		{
			d_dif_real = d_dif;
		}
		// adjust d_neg
		for(int update_iter = 0; update_iter < update_index_size-1; update_iter++)
		{
			if(d[update_iter] < 0)
			{
				d[update_iter] -= d[update_iter] * ((d_dif-d_dif_real)/d_neg);
			}
		}
	  d[update_index_size-1] = d_dif_real;
  }
	// end adjust d
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
