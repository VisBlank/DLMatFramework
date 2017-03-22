/*
Proxy template to fool the ranged for loop by returning
rbegin/rend instead of begin/end if you use the function reverse

Observe that there is 2 versions that return a normal and a const
reverse iterator.

References:
https://en.wikipedia.org/wiki/C%2B%2B11#Range-based_for_loop
http://stackoverflow.com/questions/8542591/c11-reverse-range-based-for-loop
http://kbokonseriousstuff.blogspot.co.uk/2011/09/using-reverse-iterators-with-c11-range.html
https://www.youtube.com/watch?v=V89gtNl4pZM&index=2&list=PL5jc9xFGsL8E_BJAbOw_DH6nWDxKtzBPA
https://en.wikipedia.org/wiki/Decltype
http://www.cprogramming.com/c++11/c++11-auto-decltype-return-value-after-function.html
*/
#ifndef REVERSE_RANGE_BASED_H
#define REVERSE_RANGE_BASED_H

using namespace std;

/*
    Template class that gives const iterators
    rbegin instead of begin and rend instead of end
    If you comment this your for_range will still work but will not support
    const reverse itearators
*/
template<class T>
class const_reverse_wrapper {
private:
    const T& container;
public:
  const_reverse_wrapper(const T& cont) : container(cont){ }
    /*
        decltype will infer the datatype of the expression, on c++14 this can be
        infered with auto
    */
    decltype(container.rbegin()) begin() const { return container.rbegin(); }
    //auto begin() const { return container.rbegin(); }
    decltype(container.rend()) end() const { return container.rend(); }
    //auto end() const { return container.rend(); }
};

template<class T>
const_reverse_wrapper<T> reverse(const T& cont) {
  return const_reverse_wrapper<T>(cont);
}

/*
    Template class that gives non-const iterators
    rbegin instead of begin and rend instead of end
    If you comment this your for_range will not be able to change data
*/
template<class T>
class reverse_wrapper {
private:
  T& container;
public:
  reverse_wrapper(T& cont) : container(cont){ }
    /*
        decltype will infer the datatype of the expression, on c++14 this can be
        infered with auto
    */
    decltype(container.rbegin()) begin() { return container.rbegin(); }
    //auto begin() { return container.rbegin(); }
    decltype(container.rend()) end() { return container.rend(); }
    //auto end() { return container.rend(); }
};

template<class T>
reverse_wrapper<T> reverse(T& cont) {
  return reverse_wrapper<T>(cont);
}

#endif // REVERSE_RANGE_BASED_H
