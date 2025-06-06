
"""
script start

General Outline
Nerual nets are function approximators
We can extract the function the neural net is approximating

Math statements

A matrix is transformation of a vector A matrix of directions transforming a vector is a function
This works in any dimensions of matrix and vector
if a vector is an input space N then a matrix of NxN is a eigenfunction that defines it
if a function is one to one it corresponds to these matrices
if the determinant of a matrix is 0 then the matrix reduces by at least 1 dimension
All continous functions can be writtern as a polynomial

Comp sci statments
Neural networks are an approximation of fitting the curve and finding it's minima
for a function that we are unable to define programmatically.

Feedfoward is:
Each activation is the dot product of weights and biases in respect to the previous layer 
put the nonlinearity function which acts as renormilization into the boundaries 0 and 1.

Backpropagation is:
We take the average of each training sample 
Then we take the derivative of the cost function over the weights:
(which is the chain rule of z/w * a/z * C/a)
We then apply our nonlinearity function

OUR WORKING OUT:
Idea:
We should be able to extract the functions of the nerual network mathematically

lemma:
any non linear transform is linear in a higher dimension.
proof: Hahn–Banach theorem.

explaination:
given a function F_1 and it's finite transform M,
so long as the non linear transform is continous,
in a higher dimension the transform is linear.

---
Idea: to exchange notation of a finite vector transform into a function

prop:
lets do this sandwitch method, 
we have a function F(x) it's corresponding vector would be uncountably infinite.
we have a finite vector V that corresponds to a function.
how can we approximate function F(x)?

lemma: 
spaces inbetween vector V are uncountably infinite and continous.

prop:
rearrange Feedfoward into product polynomial format:
https://en.wikipedia.org/wiki/Recurrence_relation#Linear_homogeneous_recurrence_relations_with_constant_coefficients
https://en.wikipedia.org/wiki/Linear_recurrence_with_constant_coefficients#Solution_using_generating_functions
track variables there with products

PROBLEM: activation function - relu and such uses 
how can we take this into our function

idea: the activation function cancels terms by creating a signularity
this is most notable with relu
sol: we cancel out previous terms that the activation function cancels out, and keep a record of them

so we when we Feedfoward things will be fine
prop:
we apply the same matrix of weights (function on the basis) to a new starting point(training batches),
and widen the search net of the training samples until the whole sample space is covered

idea: backpropagation is a specific case of convolution

prop: 
as the neural network is like function approximator 
it's matrix is like a conversion table between the input and output
each layer a coefficient in an unknown polynomial F(x).

lem: Backpropagation is convolution of two polynomials with the same coefficients.

idea: a taylor series approaches the polynomial, we can recover a function from this 

prop: 
smooth matrix polynomials which have finite terms
an infinite polynomial will be able to be recovered with a transform

proof: https://www.youtube.com/watch?v=zvbdoSeGAgI

Idea: Now make it within bounds

prop:
change the bounds of neural networks from 0 to 1 this is already true.
if the bounds are this we can use inverse foruier transform.
which is in this case the same as an inverse laplace

qed: inverse laplace transforms recover the function of a neural network.


Idea: mutlidim inverse fourier transforms
as all of the weights are on the real number line between 0 and 1
prop: product sum each weight vector as inputs along each layer.

Idea: how get inverse transform for each weights

ideas:
the universe is a closed space on a higher dim, a number line must be the same.
trajectoids -> a to find seperate functions in the space it took not the most efficient way of find the function but the actual path the nerual network took <- BIG LEAD
infinite weights => function local minima
BIG UPDATE - polynomial is a cauchy sequence.
matrix exponents
divergence and curl for functions => Discrete matrix transforms
borsuk ulam theorem?
e^Mt = transform / rotation
meta-networks => deriving the function from networks and then inverse fourier 
to create genralized function. Same idea as finding the transform for training batches.
eigenstates for input matrices
foruier transforms => to create human readable functions
quanta + tunneling to find absoloute minima =>

In foruier the derivative is multiplication
minibatch selectors => HYP: 
taking the largest distance inputs then getting smaller distances 
until finding convergence 
Once convergence is found for local minima i.e. det == 0 

find if convergable first?

"""

main method:

1) take the polynomial of the neural network modelled by symbolic computing
2) use inverse laplace transform to create a function modelled by that taylor series
3) this taylor series is in real domain therefore we can use inverse fourier transform
4) if this is true then this talyor series is in fourier domain 
5) the output is a system of equations each equation intersects one another 
    this can be done using linear algebra at this point
6) we then inverse fourier both the equation and output
7) we set the solver to solve this equation





