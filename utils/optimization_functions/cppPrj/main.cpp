/*
Main file

To compile:
g++ -std=c++11 main.cpp -o cppApp

Some references
http://supercomputingblog.com/openmp/tutorial-parallel-for-loops-with-openmp/
http://stackoverflow.com/questions/8384234/return-reference-to-a-vector-member-variable
https://mbevin.wordpress.com/2012/11/20/move-semantics/
*/

#define CATCH_CONFIG_RUNNER
#include "test/catch.hpp"
#define TEST false


#include "utils/tensor.h"
#include "utils/dataset.h"
#include "layers/layercontainer.h"
#include "loss/lossfactory.h"
#include "classifier/deeplearningmodel.h"
#include "utils/mathhelper.h"
#include "solver/solver.h"
#include "solver/sgd.h"
#include "layers/baselayer.h"
#include "layers/sigmoid.h"
#include "loss/crossentropy.h"

int runCatchTests()
{
    return Catch::Session().run();
}

int main() {        
    if (TEST)
    {
        return runCatchTests();
    }       
    Tensor<float> A(vector<int>({3,4}),{1,2,3,4,5,6,7,8,9,10,11,12});
    cout << "A[3x4]" << A << endl;
    A.Reshape(vector<int>({1,12}));
    cout << "A[1x12]" << A << endl;
    A.Reshape(vector<int>({12,1}));
    cout << "A[12x1]" << A << endl;
    A.Reshape(vector<int>({2,6}));
    cout << "A[2x6]" << A << endl;
    A.Reshape(vector<int>({6,2}));
    cout << "A[6x2]" << A << endl;


    return 0;
}
