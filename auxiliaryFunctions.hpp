#include<iostream>
#include<vector>
#include<math.h>
#include<stdlib.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_histogram.h>
#include<gsl/gsl_eigen.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_sort_vector.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_permutation.h>
#include<gsl/gsl_histogram.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_complex.h>
#include<gsl/gsl_sf_exp.h>
#include<gsl/gsl_sf_log.h>
#include<gsl/gsl_sf_exp.h>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_complex_math.h>
#include<gsl/gsl_multimin.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include "costFunction.hpp"

using namespace std;
void filterY( const gsl_vector * source , gsl_vector * dest, const int& pattern);

void one_vs_all(gsl_matrix * all_theta , gsl_matrix * X, gsl_vector* y, const size_t& num_classes,const int& lambda){
size_t m = X->size1; //nb of examples
size_t n = X->size2; //nb of features + 1

gsl_vector * binaryVector= gsl_vector_alloc(m);

//Initialize a Params Object, 
//containing pointers to the input  

gsl_vector * initial_theta = gsl_vector_alloc(n);//n is the number of features+1

const gsl_multimin_fdfminimizer_type *T;
T = gsl_multimin_fdfminimizer_vector_bfgs2;//gsl_multimin_fdfminimizer_conjugate_pr;
gsl_multimin_fdfminimizer * s = gsl_multimin_fdfminimizer_alloc(T, n);

double tol	= 1e-1;
double step_size= 0.1;

//setting the cost function and its gradient
gsl_multimin_function_fdf funcToMinimize;
funcToMinimize.n 	= n;
funcToMinimize.f	= &costFunction;
funcToMinimize.df 	= &costFunctionGradient;
funcToMinimize.fdf 	= &costFunctionAll;


for (int ClassIter = 1; ClassIter< num_classes+1; ClassIter++){	//ClassIter =[0..10]
	cout<< "ClassIter " <<ClassIter <<endl;

	size_t iter 	= 0;
	int status	= GSL_CONTINUE;

	filterY( y , binaryVector,ClassIter );

	Params TrainingParams(X,binaryVector);
	funcToMinimize.params 	= &TrainingParams; //pass X and y

	//intial thetas	to zeros
	gsl_vector_set_zero(initial_theta);

	gsl_multimin_fdfminimizer_set(s, &funcToMinimize, initial_theta, step_size, tol);
	
	while (status == GSL_CONTINUE && iter < 100)
	    {
		status = gsl_multimin_fdfminimizer_iterate(s);

		if (status)
			break;

		//cout<< "cost function" <<s->f <<endl;
		status = gsl_multimin_test_gradient(s->gradient, 0.5e-2);//1e-3
	
	 	if (status == GSL_SUCCESS){
			for (int i=0 ; i < n;i++){			
				gsl_matrix_set(all_theta , ClassIter-1 , i , gsl_vector_get(s->x,i));			
			}
		cout<< "Success for iter "<<ClassIter<<" min Cost is "<<s->f <<endl;
		}
	    iter++;
	    }

}//end for

gsl_multimin_fdfminimizer_free(s);
gsl_vector_free(binaryVector);
}

//***** Accuracy **//

float calculateAccuracy(gsl_matrix * all_theta, gsl_matrix * X,gsl_vector * y){
 	
int num_examples = X->size1;
int num_cols = X->size2; //nb features + 1

int nbClassifiers = all_theta->size1;

gsl_vector * maxis = gsl_vector_alloc(num_examples);
gsl_matrix * resu = gsl_matrix_alloc(num_examples,nbClassifiers);

//gsl_blas_dgemm (CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, float alpha, const gsl_matrix_float * A, const gsl_matrix_float * B, float beta, gsl_matrix_float * C)

gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,X, all_theta, 0.0, resu);

for(int i=0 ; i <num_examples ; i++){
	gsl_vector_view row = gsl_matrix_row(resu, i);
	gsl_vector_set(maxis, i , gsl_vector_max_index(&row.vector));
}

float acum=0;
for(int i=0 ; i <num_examples ; i++){
	if(gsl_vector_get(maxis, i)+1 == gsl_vector_get(y,i)){
	acum++;
	}
}

//Compare computed values with actual values, ie binary with maxis.
return 100*(acum/num_examples);
}

//******************
void filterY( const gsl_vector * source , gsl_vector * dest, const int& pattern){
int n = source->size;
	for(int i = 0 ; i<n;i++){
		if(int(gsl_vector_get(source,i)) == pattern)
			{gsl_vector_set(dest,i,1);}
		else{gsl_vector_set(dest,i,0);}
	}
}

