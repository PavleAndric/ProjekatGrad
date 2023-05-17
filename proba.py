from  tinygrad.tensor import  Tensor

x = Tensor([[1,2,3]]  ,requires_grad = True)
y  = Tensor([[5,5,5] ],requires_grad = True)
z = x+ y
y = z.sum()
#print(y.deepwalk())

t1 = Tensor([4] ,requires_grad = True)
t2 = Tensor([2], requires_grad = True)

t3 = t1.mul(t2)
t3.backward()
#print(t1.grad.numpy() , t2.grad.numpy())


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

#rom =  [x for x  in reversed(topo_sort(visited  = set(),  arr  = []))]
#print(rom)

        





