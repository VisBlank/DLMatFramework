#include <iostream>
#include <string>

using namespace std;

class BaseLayer {
protected:
	string m_name;
	int m_index;
public:
	// Pure virtual function that needs to be implemented on each layer
	virtual Tensor ForwardPropagation(const Tensor &input, const Tensor &weight, const Tensor &bias) = 0;
	virtual Tensor BackwardPropagation() = 0;
	virtual Tensor EvalBackpropNumerically(Tensor &dout) = 0;
	
	string GetName() { return m_name;};
	int GetIndex() { return m_index;};
};
