def Fibonacci(n): 
    if n<0: 
        print("Outside bounds")
    elif n==1: 
        return 0 # n==1, returns 0     
    elif n==2: 
        return 1 # n==2, returns 1
    else: 
        return Fibonacci(n-1)+Fibonacci(n-2) 
  
print(Fibonacci(9)) 