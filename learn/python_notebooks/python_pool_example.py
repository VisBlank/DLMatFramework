# Multiprocessing example, notice that this does not work on ipython
import multiprocessing

# The pool function need to be defined outside main or in another file
# It's needed to be pickeable
def  f(x):
    return x*x

# Main entry point (Could be another file)
if __name__ == '__main__':
    
    # Get number of CPUs
    cpus = multiprocessing.cpu_count()
    print('Number of cores: %d' % cpus)
    # Create a pool of execution for each core
    pool = multiprocessing.Pool(processes=cpus)
    
    # Create list with 10 elements
    list_nums = list(range(10))
    print('Original list:', list_nums)
    
    # Calculate the square of each element of the list
    x_squared = list(map(lambda n:n*n,list_nums))
    print('Calculates single-core:', x_squared)

    # Distribute each element on the pool
    x_squared_multi = pool.map(f, list_nums)
    print('Calculates multi-core:', x_squared_multi)
    
    # This does not work with lambdas
    #x_squared_multi = pool.map(lambda x: x**2, list_nums)       
