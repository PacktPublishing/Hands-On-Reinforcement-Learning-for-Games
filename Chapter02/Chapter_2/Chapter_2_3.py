fibSequence = [0,1]

def Fibonacci(n):     
    if n<0: 
        print("Outside bounds")
    elif n<= len(fibSequence): 
        return fibSequence[n-1] 
    else:   
        print("Solving for {}".format(n))
        fibN = Fibonacci(n-1) + Fibonacci(n-2)
        fibSequence.append(fibN)
        return fibN 
  
print(Fibonacci(9)) 