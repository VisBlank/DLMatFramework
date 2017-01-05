def main():
    print('Hello World\n')
    
    # Example of for loop (it will run from 0 to 2)
    for x in range(0, 3):
        print ("We're on time %d" % (x))
        
    for x in xrange(3):
        print ("We're on time %d" % (x))

    cont = 0;
    while True:        
        # If example
        if cont >= 3:
            break;
        cont += 1;
        print('Is still true');
    
# __name__ will be main if you execute from this file
if __name__ == "__main__":
    main()
