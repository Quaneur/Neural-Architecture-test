import random as r
import math
import struct as sct

class Connection:
    '''Dynamic weight class'''
    def __init__(self, mult, start_n, end_n):
        '''Connection(mult, start_n, end_n) -> Connection

mult - weight initial coefficient
start_n, end_n - from-to neurons ids for work'''
        self.begin = start_n
        self.init_mult = mult
        self.end = end_n
        self.mult = mult
        self.acc = 0.1 #Weight work accumulate
        self.accoof = 0.5 #Weight change coefficient
        self.state = True #Using or not (old optimization)
        if abs(mult) <= 0.00001:
            self.state = False
        else:
            self.state = True
    
    def Send(self, start, end):
        '''Send signal from begin neuron to end neuron'''
        end.in_d += self.mult*start.state
        end.c += 1
        self.acc = self.acc*self.accoof+start.state*0.1

    def Reset(self):
        '''Reset weight to initial condition'''
        self.acc = 0
        self.mult = self.init_mult       
    
    def Mutate(self, mutate_param: float):
        '''Randomly change the weight with coefficient'''
        self.init_mult = self.init_mult+(r.random()*2-1)*mutate_param
        self.accoof = max(0, min(self.accoof+(r.random()*2-1)*mutate_param, 1))
        if abs(self.mult) <= 0.00001:
            self.state = False
        else:
            self.state = True

    def Update(self, good_coof_delta):
        '''Change weight on good coefficient change'''
        presave = float(self.mult)
        self.mult = self.mult+self.mult/abs(self.mult)*(self.acc*good_coof_delta)
        self.mult /= 1+math.log10(abs(self.mult))/2
        if math.isnan(self.mult):
            print(f"Warning! Weight overflow ({self.begin}->{self.end})!")
            self.mult = presave
        #print(self.mult)
        if abs(self.mult) <= 0.00001:
            self.state = False
        else:
            self.state = True
    
    def to_bytes(self):
        '''Weight to bytes convertor'''
        data = sct.pack("<HHffff", self.begin, self.end, self.init_mult, self.mult, self.acc, self.accoof)
        return data
    
    def from_bytes(self, data):
        '''bytes to Weight convertor'''
        
        b = sct.unpack("<HHffff", data)
        self.begin = b[0]
        self.end = b[1]
        self.init_mult = b[2]
        self.mult = b[3]
        self.acc = b[4]
        self.accoof = b[5]

    def __repr__(self):
        return f"<Connection>:b:{self.begin.id};n:{self.end.id};mult:{self.mult}"