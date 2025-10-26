# Neural-Architecture-test

Just simple test library of neural architecture with dynamic weights on Python (Still in work)

`Connections.py` - The Connection class file.

`Neurons.py` - The main library class. For using example, examine this file.

Just add this files into your project and import `Neurons.py` in your executable.

For using GPU acceleration, you need to has installed `pyopencl` library and instead `Brain` class use `GPU_Brain`.

Saving for `GPU_Brain` is not working now.

## How to use?

It's really simple. There's an example for understanding:
```python
import Neurons as nr # Importing a lib.
import numpy as np # For generating input database 

inputs = 3 # Input neurons count
inners = 8 # Inner neurons count
outputs = 3 # Output neurons count

data_c = 300 # Count for input data
inp = np.random((100, inputs)) # Generating an input data that will pushed into our network

c = inputs+inner+1+outputs # Summary neurons count (including neuron of good)

brain = nr.Brain(c, inputs, outputs, nr.sigmoid) # Creating a neural network.

# Because in this example we will use not evolved neural network,
# we need to mutate it for normal work of dynamic weighs.
brain.Mutate(0.5)

for i, x in enumerate(list(inp)):
    print(f"step {i}: inputs{list(x)}")
    brain.setInput(list(x)) # Setting an input neurons
    brain.Update() # Updating the neural network state
    outs = brain.getOutput() # And getting outputs.
    print(f"Outputs: {outs}\n")

# If you want to save it, then:
with open("example.NN", "wb") as f: # Open/create the binary file (recommended '.NN' type of file)
    data = brain.to_bytes() #Converting brain to bytes
    f.write(data) # And writing into file.
```
## Why this library has been created?
Well, if be honest, idk. Just some randomly generated idea.

But after some think, i understanded that this type of neural architectuce can be useful adn decided public what i'm coded.

# About architecture
In short explaination: This architecture is just chaotic mass of dynamic conections and neurons that will somewhat work.
## What the '*dynamic*' weights/connections?
All of current popular neural networks (GPT 5, for example) having inside a billions of weights that i'll call *stable connections* (because they're can't change while work).

This architecture using dynamic weights/connections, that changing in neural network's work process.
Instead of having one value of weight, connection has a 3 number (not including initial value):
1) Currennt weight
2) Work value (how actively this weight is using)
3) Change coefficient (the amount of change, based on good coefficient)

The good coefficient also changing, and it's value depenced on value of good neuron (one of output neurons).
