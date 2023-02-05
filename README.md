So...
# neural-network-on-python
This my neural network to predict numbers by MINIST dataset.<br>
To know how to launch and see how it work, check next.

<h1>How Neural network work?</h1>
<h3>Layers</h3>
Network has 1 input layer, 2 hidden layers and output layer.
<ul>
<li>Input layer - 784(There are pixels of photo)</li>
<li>First hidden layer - 128</li>
<li>Second hidden layer - 64</li>
<li>Output layer - 10</li>
</ul>

<h3>Functions</h3>
I used 2 activation functions: <b>softmax</b> and <b>sigmoid</b><br><br>
Sigmoid: <b>1/(1 + np.exp(-x))</b><br>
And of course softmax: (this function doesn`t have graph) <b>np.exp(x - x.max()) / np.sum(exps, axis=0)</b>
<br>
Sigmoid is used in two layers (input and first hidden)

<h1>How launch it</h1>

You must write to console: `python main.py`<br>
And the algorithms (backpropagation) will start working.<br>
Also I have first(bad) version who doesn`t work.<br>
If you want you can fix it)
