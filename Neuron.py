from Connection import *
import math
import time
import struct as sct
import random as r
try:
    import os
    os.environ["PYOPENCL_CTX"] = "0"
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
    os.environ["PYOPENCL_NO_CASHE"] = "y"
    del os
    import numpy as np
    import pyopencl as cl
    _GPUACCEL = True
    with open("GPU.cpp", "r") as f:
        _gpu_soft = f.read()
        print(_gpu_soft)
    _gpu_context = cl.create_some_context()
    _gpu_queue = cl.CommandQueue(_gpu_context)
    _gpu_functs = cl.Program(_gpu_context, _gpu_soft).build()
    '''_gpu_update1 = _gpu_functs.Update1#cl.Kernel(_gpu_functs, "Update1")
    _gpu_update2 = _gpu_functs.Update2#cl.Kernel(_gpu_functs, "Update2")
    _gpu_update3 = _gpu_functs.Update3#cl.Kernel(_gpu_functs, "Update3")
    _gpu_mutate_c =_gpu_functs.MutateC# cl.Kernel(_gpu_functs, "MutateC")
    _gpu_mutate_n =_gpu_functs.MutateN# cl.Kernel(_gpu_functs, "MutateN")
    _gpu_random =  _gpu_functs.GenRand# cl.Kernel(_gpu_functs, "GenRand")'''
    mf = cl.mem_flags

except ModuleNotFoundError as e:
    print(f"<{__name__}> |WARNING: 'pyopencl' library is not found. GPU software acceleration is disabled.\n|Install this library for using GPU acceleration.")
    _GPUACCEL = False

def sigmoid(x, c):
    '''Sigmoid function'''
    return 1/(1+math.exp(-x))

def linear(x, c):
    '''Linear function'''
    return 0.5+x/c/2

def softrelu(x, c):
    '''Pseudo-soft RELU function'''
    state = ((x/c>1)<<1)|(x/c<0)
    d = 0
    match state:
        case 0: #0<=x<=1:
            d = x/c
        case 1: #x<0:
            d =  x/c/10
        case 2: #x>0:
            d = 1+(x/c-1)/10
    return 0.5+d/2

def relu(x, c):
    '''Basic RELU function'''
    return 0.5+max(0, min(x/c, 1))/2

def blh(x):
    '''Binary list to number convert'''
    res = 0
    for i in range(len(x)):
        res += x[i]*2**i
    return res

class Neuron:
    '''The neuron. Using in `Brain` class'''
    def __init__(self, typen: int = 0):
        self.act_funct = None
        self.state = False
        self.in_d = 0.0
        self.c = 0
        self.id = 0
        self.type = typen #0 - normal; 1 - input; 2 - output;
        
    def UpdateState(self):
        '''Uptade the state of neuron.'''
        if self.type == 1:
            return
        try:
            self.state = self.act_funct(self.in_d, self.c)
        except BaseException as e:
            print(f"Neuron {self.id} has a error: {e}")
            self.state = 0
        self.in_d = 0
        self.c = 0

    def to_bytes(self):
        '''Neuron to bytes convertation'''
        data = sct.pack("<HfBB", self.id, self.state, self.type, 0)
        return data

    if _GPUACCEL:
        def to_np_bytes(self):
            _t = [linear, relu, softrelu, sigmoid]
            c = _t.index(self.act_funct)
            j = (self.type&3)|((c<<2)&15)
            data = sct.pack("<HBxffHxx", self.id, j, self.in_d, self.state, self.c)
            return data
        pass

        def from_np_bytes(self, data):
            _t = [linear, relu, softrelu, sigmoid]
            l = sct.unpack("<HBxffHxx", data)
            self.id = l[0]
            self.type = l[1]&3
            self.act_funct = _t[(l[1]>>2)&15]
            self.in_d = l[2]
            self.state = l[3]
            self.c = l[4]


    def from_bytes(self, data):
        '''Bytes to Neuron convertation'''
        b = sct.unpack("<HfBB", data)
        self.id = b[0]
        self.state = b[1]
        self.type = b[2]
        self.act_funct = sigmoid

    def Reset(self):
        '''Brain reset callback'''
        self.state = 0
        
    def Mutate(self, coof: float):
        '''Brain 'Mutate' callback'''
        pass
        
    def __str__(self):
        return f"<Neuron:{self.state}>"

class Brain:
    '''The main brain class.
Initialize: Brain([summary_n_count], [n_input_count], [n_output_count])'''
    def __init__(self, neur_c, inputs = 0, outputs = 0, act_funct=sigmoid):
        neurons = []
        connections = []
        input_n = []
        output_n = []
        neur_c = neur_c+1
        outputs = outputs+1
        for i in range(neur_c):
            n = Neuron()
            n.act_funct = act_funct
            if i < inputs:
                n.type = 1
                input_n.append(n)
            elif i > neur_c-1-outputs:
                n.type = 2
                output_n.append(n)
            n.act_funct = sigmoid
            n.id = i
            neurons.append(n)
            for j in range(neur_c):
                if (i^j) and (j>=inputs) and (i<=neur_c-1-outputs):
                    c = Connection(r.random()*2-1, i, j)
                    connections.append(c)

        
        self.c = neur_c
        self.neurons = neurons
        self.connections = connections
        self.good = 0
        self.last_good = 0
        self.input_n = input_n
        self.output_n = output_n
        
    def update(self):
        '''Update the Brain to next state'''
        max_d = -1
        for c in self.connections:
            c.Send(self.neurons[c.begin],self.neurons[c.end])
        for n in self.neurons:
            max_d = max(max_d, abs(n.in_d))
            n.UpdateState()
        print(self.output_n[0].state)
        self.last_good = self.good
        self.good = self.good*0.995+self.output_n[0].state*0.005
        delta = (self.good-self.last_good)*250
        print(delta)
        print(max_d)
        for c in self.connections:
            c.Update(delta)

                   
    def GetState(self):
        '''Get current state'''
        res = []
        for n in self.neurons:
            res.append(n.state)
        return res

    def setInput(self, input_list: list):
        '''Set input neurons states.
Input data - number list that >= count of input neurons'''
        if len(input_list) < len(self.input_n):
            raise TypeError(f"Input data less than needed ({len(input_list)}<{len(self.input_n)})")
        for i in range(len(self.input_n)):
            self.input_n[i].state = input_list[i]

    def ResetState(self):
        '''Set all neurons state to zero'''
        for n in self.neurons:
            n.state = False
        
    def getOutput(self):
        '''Get output as list of float32 numbers'''
        return [self.output_n[i].state for i in range(len(self.output_n))]

    def Reset(self):
        '''Reset Brain to initial state (including weights)'''
        for n in self.neurons:
            n.Reset()
        for c in self.connections:
            c.Reset()

    def to_bytes(self):
        '''Convert Brain object to bytes'''
        head = bytearray(sct.pack("<ffHQ", self.last_good, self.good, len(self.neurons), len(self.connections)))
        for n in self.neurons:
            head += n.to_bytes()
        for c in self.connections:
            head += c.to_bytes()
        return head

    def from_bytes(self, data):
        '''Convert bytes object to Brain class'''
        self.neurons = []
        self.input_n = []
        self.output_n = []
        self.connections = []
        head = sct.unpack("<ffHQ", data[:18])
        self.c = head[2]
        self.last_good = head[0]
        self.good = head[1]
        n_count = head[2]
        c_count = head[3]
        offset_data = 18
        for ni in range(n_count):
            n = Neuron(0, 0, 0)
            n.from_bytes(data[offset_data:offset_data+8])
            self.neurons.append(n)
            match n.type:
                case 1:
                    self.input_n.append(n)
                case 2:
                    self.output_n.append(n)
                case 0:
                    pass
            offset_data += 8
        for ci in range(c_count):
            c = Connection(0, 0, 0)
            c.from_bytes(data[offset_data:offset_data+20])
            self.connections.append(c)
            offset_data += 20

    def Clone(self):
        '''Clone brain'''
        new_brain = Brain(0)
        new_brain.c = self.c
        new_ns = []
        new_cs = []
        inp_n = []
        out_n = []
        for n in self.neurons:
            nn = Neuron(n.type)
            nn.id = n.id
            nn.act_funct = n.act_funct
            nn.state = n.state
            new_ns.append(nn)
            if nn.type == 1:
                inp_n.append(nn)
            elif nn.type == 2:
                out_n.append(nn)
        for c in self.connections:
            nc = Connection(0, c.begin, c.end)
            nc.init_mult = c.init_mult
            nc.mult = c.mult
            nc.acc = c.acc
            nc.accoof = c.accoof
            nc.state = c.state
            new_cs.append(nc)


        new_brain.neurons = new_ns
        new_brain.connections = new_cs
        new_brain.input_n = inp_n
        new_brain.output_n = out_n
        return new_brain

    def Mutate(self, coof):
        '''Randomly change the weights and neurons'''
        for n in self.neurons:
            n.Mutate(coof)
        for c in self.connections:
            c.Mutate(coof)

if _GPUACCEL:
    class GPU_Brain:
        def __init__(self, neur_c, inputs = 0, outputs = 0, act_funct=sigmoid):
            neurons = []
            connections = []
            input_n = []
            output_n = []
            neur_c = neur_c+1
            outputs = outputs+1
            for i in range(neur_c):
                n = Neuron()
                n.act_funct = act_funct
                if i < inputs:
                    n.type = 1
                    input_n.append(n)
                elif i > neur_c-1-outputs:
                    n.type = 2
                    output_n.append(n)
                n.act_funct = sigmoid
                n.id = i
                neurons.append(n)
                for j in range(neur_c):
                    if (i^j) and (j>=inputs) and (i<=neur_c-1-outputs):
                        c = Connection(r.random()*2-1, i, j)
                        connections.append(c)

            data1 = bytearray()
            data2 = bytearray()
            for n in neurons:
                data1 += n.to_np_bytes()
            for c in connections:
                data2 += c.to_bytes()

            self.c = neur_c
            self.inputs = inputs
            self.outputs = outputs
            self.con_count = len(connections)
            self.out_offs = neur_c-outputs
            self.inner_offs = inputs
            self.neurons_d = np.array(list(data1), dtype=np.uint8)
            self.connections_d = np.array(list(data2), dtype=np.uint8)
            self.good = 0
            self.last_good = 0

            self.queue = _gpu_queue
            self.n_buff = cl.Buffer(_gpu_context, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=self.neurons_d)
            self.c_buff = cl.Buffer(_gpu_context, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=self.connections_d)

        def copy(self):
            o = GPU_Brain(0)
            o.c = self.c
            o.input_n = self.input_n
            o.output_n = self.output_n
            o.out_offs = self.out_offs
            o.inner_offs = self.inner_offs
            o.good = self.good
            o.last_good = self.last_good
            o.con_count = self.con_count
            o.neurons_d = self.neurons_d.copy()
            o.connections_d = self.coonections_d.copy()

            o.queue = self.queue
            o.n_buff = cl.Buffer(_gpu_context, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=o.neurons_d)
            o.c_buff = cl.Buffer(_gpu_context, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=o.connections_d)
            return o

        def update(self):
            _gpu_functs.Update1(self.queue, (self.con_count,), None, self.c_buff, self.n_buff)
            _gpu_functs.Update2(self.queue, (self.c,), None, self.n_buff)
            cl.enqueue_copy(self.queue, self.neurons_d, self.n_buff)
            goodn = Neuron()
            n_data = bytes(list(self.neurons_d[self.out_offs*16:self.out_offs*16+16]))
            goodn.from_np_bytes(n_data)
            self.last_good = self.good
            self.good = self.good*0.995+goodn.state*0.005
            delta = (self.good-self.last_good)*250
            _gpu_functs.Update3(self.queue, (self.con_count,), None, self.c_buff, np.float32(delta))
            cl.enqueue_copy(self.queue, self.connections_d, self.c_buff)

        def Mutate(self, coof):
            brgn = r.getrandbits(64)
            _gpu_mutate_c(self.queue, (self.con_count,), None, self.c_buff, np.uint64(brgn), np.float32(coof))
            ergn = r.getrandbits(64)
            _gpu_mutate_n(self.queue, (self.c,), None, self.n_buff, np.ulonglong(ergn), np.float32(coof))
                                
        def GetState(self):
            '''Get current state'''
            res = []
            for offs in range(0, self.neurons_d.nbytes, 16):
                n = Neuron()
                n.from_np_bytes(bytes(list(self.neurons_d[offs:offs+16])))
                res.append(n.state)
                del n
            return res

        def getOutput(self):
            res = []
            for offs in range(self.out_offs+16, self.neurons_d.nbytes, 16):
                n = Neuron()
                n.from_np_bytes(bytes(list(self.neurons_d[offs:offs+16])))
                res.append(n.state)
                del n
            return res

        def setInput(self, inputs):
            if len(inputs) < self.inputs:
                raise TypeError(f"Input data less than needed ({len(inputs)}<{self.inputs})")
            cl.enqueue_copy(self.queue, self.neurons_d, self.n_buff)
            for inp in range(self.inputs):
                data = self.neurons_d[inp*16:inp*16+16]
                n = Neuron()
                n.from_np_bytes(bytes(list(data)))
                n.state = inputs[inp]
                data = n.to_np_bytes()
                self.neurons_d[inp*16:inp*16+16] = np.array(list(data), np.uint8)
            cl.enqueue_copy(self.queue, self.n_buff, self.neurons_d)


'''test = np.ndarray((1000,), np.float32)
test_buff = cl.Buffer(_gpu_context, mf.READ_WRITE, test.nbytes)
seed = r.getrandbits(64)
_gpu_functs.GenRand(_gpu_queue, (1000,), None, test_buff, np.ulonglong(seed))
cl.enqueue_copy(_gpu_queue, test, test_buff)
print(test)
exit()'''
'''Basic perfomance and functions test.'''
if __name__ == "__main__":
    from PIL import Image
    inp = 5
    outp = 5
    count = 256

    brain = GPU_Brain(inp+outp+1+count, inp, outp, linear) #Initializing Brain with 60 inner neurons, 100 inputs and 5 outputs + 1 good neuron
    #brain.Mutate(0.5) #Randomly change the Brain

    #Save test
    '''with open("test_save.NN", "wb") as f:
        data = brain.to_bytes()
        f.write(data)'''

    #Clone speed perfomance
    '''s_clone = time.time()
    cloned = brain.Clone()
    clone_t = time.time()-s_clone
    print(f"Clone speed:{clone_t*1000}ms")'''

    #Brain work test
    step = 0
    history = []
    run = True
    ltime = time.time()
    print("Begin")
    print('---------------------------')
    try:
        for i in range(4096*4):
            print(f"Step:{i+1}")

            brain.setInput([r.random() for _ in range(5)]) #Setting some inputs
            brain.update() #Updating the brain state
            print(f"GoodCoof: {brain.good}")
            d = brain.GetState() #Getting neurons state (for image graph)
            history.append(d)
            upd_time = round((time.time()-ltime)*1000)
            ltime = time.time()
            print(f"Update_time: {upd_time}ms")
            print('---------------------------')
    except KeyboardInterrupt:
        pass
        
    #Saving brain activity history
    img = Image.new("RGB", (len(history[0]), len(history)))
    p = 0
    l1 = len(history[0])
    for d in history:
        
        step = history.index(d)
        for n in d:
            img.putpixel((p%l1, p//l1), (max(0, min(round(50*((p%l1<inp)or(p%l1==brain.c-brain.outputs))+205*n*((p%l1<inp)or(p%l1==brain.c-brain.outputs))), 255)), max(0, min(round(50*((l1-outp>p%l1>=inp))+205*n*((l1-outp>p%l1>=inp))), 255)), max(0, min(round(50*(l1-outp<=p%l1)+205*n*(l1-outp<=p%l1)), 255))))
            p+=1
    img.save("hm.png")
