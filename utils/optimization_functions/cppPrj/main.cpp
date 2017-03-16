/*
Main file

To compile:
g++ -std=c++11 main.cpp -o cppApp

Some references
http://supercomputingblog.com/openmp/tutorial-parallel-for-loops-with-openmp/
http://stackoverflow.com/questions/8384234/return-reference-to-a-vector-member-variable
https://mbevin.wordpress.com/2012/11/20/move-semantics/
*/
#include "utils/tensor.h"


int main() {    
    Tensor<float> A(vector<int>({2,2}));
    A(0,0) = 1;
    A(0,1) = 2;
    A(1,0) = 3;
    A(1,1) = 4;
    Tensor<float> B(vector<int>({2,2}));
    B(0,0) = 5;
    B(0,1) = 6;
    B(1,0) = 7;
    B(1,1) = 8;

    cout << "Matrix A:" << endl;
    A.print();
    cout << "Matrix B:" << endl;
    B.print();

    Tensor<float>C = A*B; // C.assign(A.mult(B));
    cout << "A*B" << endl;
    C.print();

    Tensor<float>D = A+B;
    cout << "A+B" << endl;
    D.print();

    Tensor<float>E = A-B;
    cout << "A-B" << endl;
    E.print();

    Tensor<float>F = B;
    cout << "F=B" << endl;
    F.print();

    Tensor<float>G = A*(float)2.5;
    cout << "G=A*2.5" << endl;
    G.print();

    Tensor<float>H = A+(float)1.1;
    cout << "H=A*1.1" << endl;
    H.print();

    return 0;
}
