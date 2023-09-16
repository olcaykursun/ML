#Run the whole program once expecting and getting "2 2" as output
#Don't run it cell by cell, just run it once (hit F5)
#Then go to console and type n=3

#Now that we changed n to 3, calling foo should output "3 3", but we get "3 2"

#Approach 1: Stay in console and type foo(n)  
#Approach 2: Run the second code cell

n = 2

def foo(x):
    print(x, n)

#%%
foo(n)
