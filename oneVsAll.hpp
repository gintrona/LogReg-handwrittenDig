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

//** filterY declaration
void filterY( const gsl_vector * source , gsl_vector * dest, const int& pattern);

//Calculate the all_theta matrix, where each row is a class and each column represents a feature
void one_vs_all(gsl_matrix * all_theta , gsl_matrix * X, gsl_vector* y, const size_t& num_classes){

size_t m = X->size1; //nb of examples
size_t n = X->size2; //nb of features + 1

gsl_vector * binaryVector= gsl_vector_alloc(m);

gsl_vector * initial_theta = gsl_vector_alloc(n);//n is the number of features+1

const gsl_multimin_fdfminimizer_type *T;

//** choosing the minimizer
T = gsl_multimin_fdfminimizer_vector_bfgs2;	
gsl_multimin_fdfminimizer * s = gsl_multimin_fdfminimizer_alloc(T, n);

double tol	= 1e-1;
double step_size= 1.0;

//** setting the cost function and its gradient
gsl_multimin_function_fdf funcToMinimize;
funcToMinimize.n 	= n;
funcToMinimize.f	= &costFunction;
funcToMinimize.df 	= &costFunctionGradient;
funcToMinimize.fdf 	= &costFunctionAll;


for (int ClassIter = 1; ClassIter< num_classes+1; ClassIter++){//ClassIter = [1..10]
	cout<< "ClassIter " <<ClassIter <<endl;

	size_t iter 	= 0;
	int status	= GSL_CONTINUE;

	//** make a binary vector from the 
	filterY( y , binaryVector,ClassIter );

	//** Initialize a Params Object, containing pointers to the input  
	Params TrainingParams(X,binaryVector);
	funcToMinimize.params 	= &TrainingParams; //pass X and y

	//** intial thetas to zeros
	gsl_vector_set_zero(initial_theta);

	gsl_multimin_fdfminimizer_set(s, &funcToMinimize, initial_theta, step_size, tol);
	
	while (status == GSL_CONTINUE && iter < 100)
	    {
		status = gsl_multimin_fdfminimizer_iterate(s);

		if (status)
			break;

		status = gsl_multimin_test_gradient(s->gradient, 0.5e-2);//1e-3
	
	 	if (status == GSL_SUCCESS){
			for (int i=0 ; i < n;i++){			
		 gsl_matrix_set(all_theta , ClassIter-1 , i , gsl_vector_get(s->x,i));			
			}
		cout<< "Success for iter "<<ClassIter<<" min Cost is "<<s->f <<endl;
		}
	    iter++;
	    }
}
gsl_vector_free(initial_theta);
gsl_multimin_fdfminimizer_free(s);
gsl_vector_free(binaryVector);
}

//** Accuracy function **//

float calculateAccuracy(gsl_matrix * all_theta, gsl_matrix * X,gsl_vector * y){
 	
size_t num_examples = X->size1;
size_t num_cols = X->size2; //nb features + 1

size_t nbClassifiers = all_theta->size1;

gsl_vector * maxis = gsl_vector_alloc(num_examples);
gsl_matrix * resu = gsl_matrix_alloc(num_examples,nbClassifiers);

/*Compute resu = X all_theta' ; all_theta is transponse */
gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0,X, all_theta, 0.0, resu);

//vector maxis contains the index of the maximum value for each row of resu
for(size_t i=0 ; i <num_examples ; i++){
	gsl_vector_view row = gsl_matrix_row(resu, i);
	gsl_vector_set(maxis, i , gsl_vector_max_index(&row.vector));
}

float acum=0;
for(size_t i=0 ; i <num_examples ; i++){
	if(gsl_vector_get(maxis, i)+1 == gsl_vector_get(y,i)){
	acum++;
	}
}

//Compare computed values with actual values, ie binary with vector maxis.
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

//******************
void loadInput(string nameOfFile,gsl_matrix *TrainingX , gsl_vector *TrainingY ){

ifstream myfile(nameOfFile);

//AUX VARIABLES
int row(0);
string line;	
vector<string>::iterator it;

if((myfile).is_open())
	{while(getline(myfile,line))
	  {int col=0;

	    	istringstream buffer(line);
	    	istream_iterator<string> beg(buffer);
		//** default constructor end-of-stream
		istream_iterator<string> end;
		
		//** range constructor
	    	vector<string> tokens(beg, end);
	    	vector<string>::iterator it_last = std::prev(tokens.end());

		//** first column filled with 1
		gsl_matrix_set(TrainingX , row, 0, 1.0); 

		for(it = tokens.begin(),col=1; it != it_last; ++it,col++){
			//** read the number and convert it to a float //better way?
			gsl_matrix_set(TrainingX , row, col, atof((*it).c_str()));
		}
		//** last column is the input Y
		gsl_vector_set(TrainingY, row, atof((*it_last).c_str()));
		row++;	  
	  }
	 (myfile).close();
	}
}
