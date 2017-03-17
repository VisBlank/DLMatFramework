/*
    Loss factory
    References:
    http://stackoverflow.com/questions/30774267/whats-a-good-way-of-factorys-create-method-in-c11
    http://bobah.net/d4d/source-code/misc/factory-cxx11
*/
#ifndef LOSSFACTORY_H
#define LOSSFACTORY_H
#include <string>
#include <memory>
#include "baseloss.h"
#include "crossentropy.h"
#include "multiclasscrossentropy.h"

using namespace std;
template <typename T>
class LossFactory
{
public:
    // No constructor
    LossFactory() = delete;
    static shared_ptr<T> GetLoss(){
        return make_shared<T>();
    }

};

#endif // LOSSFACTORY_H
