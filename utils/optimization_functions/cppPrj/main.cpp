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
#define TEST true


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
    if (TEST){
        return runCatchTests();
    }

    return 0;
}
