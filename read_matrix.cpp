#include "auxiliaryFunctions.hpp"

using namespace std;


int main(int argc,char *argv[]){

// Setup the parameters you will use for this part of the exercise
size_t input_layer_size  = 400;  // 20x20 Input Images of Digits
size_t number_of_classes  = 10;          // 10 labels, from 1 to 10   
                          	 //(note that we have mapped "0" to label 10)

/* =========== Part 1: Loading and Visualizing Data =============
  The dataset contains handwritten digits.
*/

//DEFINE THE SIZE OF THE TRAINING SET
size_t numExamples = 5000;
size_t numFeatures = 400;
gsl_matrix * all_theta = gsl_matrix_alloc(number_of_classes, numFeatures+1);

{  
     FILE * f = fopen ("resu.dat", "rb");
     gsl_matrix_fread (f, all_theta);
     fclose (f);
  }

for (int i=0 ; i<number_of_classes;i++){
	for (int j=0 ; j<numFeatures+1;j++){
cout<<gsl_matrix_get(all_theta,i,j)<<" ";
} cout<<endl;}


gsl_matrix_free(all_theta);

// Calculate the accuracy by comparing the input value and the
// predicted value for each training item
//calculateAccuracy(all_theta, X, y);

return 0;
}
