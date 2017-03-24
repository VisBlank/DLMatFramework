/*
    Class used to implement range of values
    References:
    http://en.cppreference.com/w/cpp/algorithm/iota
    http://stackoverflow.com/questions/7185437/is-there-a-range-class-in-c11-for-use-with-range-based-for-loops
*/
#ifndef RANGE_H
#define RANGE_H

#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;

template <typename T>
class range{
private:
    vector<T> m_buffer;
    int m_begRange;
    int m_endRange;
public:
    range() = delete;
    range (const int &beg, const int &end):m_begRange(beg), m_endRange(end){
        if ((end >= 0) && (beg >= 0)){
            if (end <= beg){
                throw invalid_argument("range(end) should be bigger than ramge(begining).");
            }
            m_buffer = vector<T>((end+1)-beg,0);
            iota(m_buffer.begin(), m_buffer.end(), beg);
        } else {
            m_buffer.clear();
        }
    }
    // Iterators to manipulate the vector class member
    typename std::vector<T>::iterator begin(){return m_buffer.begin();}
    typename std::vector<T>::iterator end(){return m_buffer.end();}
    typename std::vector<T>::const_iterator begin() const {return m_buffer.begin();}
    typename std::vector<T>::const_iterator end() const {return m_buffer.end();}

    int Min() const { return m_begRange;}
    int Max() const { return m_endRange;}
    bool empty() const {return m_buffer.empty();}
    size_t size() const {return m_buffer.size();}

    // Overload the << Operator to use this class with cout
    // https://msdn.microsoft.com/en-us/library/1z2f6c2k.aspx
    friend ostream& operator<<(ostream& os, const range<T>& right){
        stringstream str_stream;
        if (!right.empty()){
            for_each(right.m_buffer.begin(), right.m_buffer.end(), [&str_stream](T m){str_stream << m << " ";});
        } else {
            str_stream << "empty []";
        }
        os << str_stream.str();
        return os;
    }
};

#endif // RANGE_H
