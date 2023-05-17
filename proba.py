graph = {
    "1": [],
    "3": ["1"],
    "2": ["3"],
    "5": ["2" , "0"],
    "0": [], 
    "4": ["0" , "1"]}

# topo je   5 4 3 2 1 0
# ili       4 5 2 3 1 0
# topo sort
# 4 5 0 2 3 1 
# 4 -> 0 ,1 
# 5 -> 2,0 
# 0 -> null
# 2 -> 3 
# 3 -> 1
# 1 -> null

def topo_sort(visited , arr):
    def topo(node_):
        if node_ not in visited:
            visited.add(node_)
            for x_ in graph[node_]:
                topo(x_)
            arr.append(node_)
    for node in graph:
        topo(node)
    return arr

from tinygrad.tensor import  Tensor as ts

t1  = ts([1.] , requires_grad= True)
t2 = ts([5.] , requires_grad = True)
t4 = ts([4.])

t3 = t1 + t2
t5 = t3 * t4 
t5.backward()

print(t1.grad.numpy() , t2.grad.numpy(), t3.grad.numpy())


