from  tinygrad.tensor import  Tensor

x = Tensor([[1,2,3]]  ,requires_grad = True)
y  = Tensor([[5,5,5] ],requires_grad = True)
z = x+ y
y = z.sum()
#print(y.deepwalk())


t1 = Tensor([1] , requires_grad = True)
t2 = Tensor([2] , requires_grad = True)

t3 = t1 + t2

#print(type(z._ctx))
#print(t3._ctx , "romcina")
#for x  in t3.deepwalk():
#    print(x)
#    print(x._ctx)

#class ROM:
#    def __init__(self, ime):
#        self.ime = ime
#
#
#rom = ROM("esketitit")
#
#setattr(rom , "ctx" , 5)
#
#print( "haha", rom.ctx)

class proba:
    def __init__(self, ime , prezime):
        self.ime = ime
        self.prezime = prezime
    @staticmethod
    def print_ime_prez(x,y = None):
        return x + y if y is not None else x
    
def sterr(ime):
    setattr(proba , "ime" , ime)

#pro = proba("ide" , "gas")
#print(pro.print_ime_prez("rom " , "todor"))
#sterr("laki")
#print(pro.print_ime_prez("pavle " ))

import functools

class Cell:
    def __init__(self):
        self._alive = False
        self.ctx = None
    @property
    def alive(self):
        return self._alive
    def set_state(self, state):
        self._alive = bool(state)
    set_alive = functools.partialmethod(set_state, True) # set_alive je seter alive
    set_dead = functools.partialmethod(set_state, False) # set_dead je seter dead 

niz = [1,2,3]
def add(broj):
    broj += 1 
    return broj
print([x  for x in niz]) 
# niz2 = add(*[x  for x  in niz])