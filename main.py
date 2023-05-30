from manim import (Tex, Scene, 
                    VGroup, Write,
                    FadeIn, FadeOut)


import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy as sp
import sympy as syp

"""
script start

General Outline
Nerual nets are function approximators
We can extract the function the neural net is approximating

Math statements

A matrix is transformation of a vector A matrix of directions transforming a vector is a function
This works in any dimensions of matrix and vector
if a vector is an input space N then a matrix of NxN is a eigenfunction that defines it
if a function is one to one it corresponds to these matricies
if the determinant of a matrix is 0 then the matrix reduces by at least 1 dimension
All continous functions can be writtern as a polynomial

Comp sci statments
Neural networks are an approximation of fitting the curve and finding it's minima
for a function that we are unable to define programmatically.

Feedfoward is:
Each activation is the dot product of weights and baises in respect to the previous layer 
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

PROBLEM: sigmoid function - relu and such uses 
prop:
we apply the same matrix of weights (function on the basis) to a new starting point(training batches),
and widen the search net of the training samples until the whole sample space is covered

idea: backpropagation is a specific case of convolution

prop: 
as the neural network is like function approximator 
it's matrix is like a conversion table between the input and output
each layer a coefficient in an unknown polynomial F(x).

lem: Backpropagation is convolution of two polynomials with the same coefficients.

idea: polynomial approaches a taylor series, we can recover a function from this 
lem: use the 

prop: 
smooth matrix polynomials which have finite terms
an infinite polynomial will be able to be recovered with a transform



proof: https://www.youtube.com/watch?v=zvbdoSeGAgI

Idea: Now make it within bounds

prop:
change the bounds of neural networks from 0 to 1 this is already true.
if the bounds are this we can use foruier transform.
which is an inverse laplace

qed: inverse fourier transforms recover the function of a neural network.

Idea: mutlidim fourier transforms

prop: product sum each weight vector as inputs along each layer.




ideas:
infinite weights => function local minima
BIG UPDATE - polynomial is a cauchy sequence.
matrix exponents
divergence and curl for functions => Discrete matrix transforms
borsuk ulam theorem?
e^Mt = transform / rotation
meta-networks => deriving the function from networks and then inverse fourier 
to create genralized function. Same idea as finding the transform for training batches.
IS FOURIER TRANSFORM A RED HERRING?
eigenstates for input matricies
foruier transforms => to create human readable functions
quanta + tunneling to find absoloute minima =>

minibatch selectors => HYP: 
taking the largest distance inputs then getting smaller distances 
until finding convergence 
Once convergence is found for local minima i.e. det == 0 

find if convergable first?

"""

def get_poly_eqs(layers_total):
    # setup symbols for computation
    weight= syp.Symbol("w_0")
    activation = syp.Symbol("a")
    bias = syp.Symbol("b_0")
    sigmoid = syp.Function("sig")

    # create neruonal activation equations
    staring_eq = syp.Eq(sigmoid((weight * activation) + bias))
    eq_system = [staring_eq]

    # summate equations to get output
    for i in range(1, layers_total):
        eq_system.append(sigmoid(syp.Symbol("w_" + str(i)) * eq_system[-1]) + 
                         syp.Symbol("b_" + str(i)) + eq_system[-1])
    
    return eq_system

def subst_into_system(fft_layers, eq_system):
    for i in range(0, fft_layers.len()):
        weight_symbol = syp.Symbol("w_" + str(i))
        bias_symbol = syp.Symbol("b_" + str(i))

        # fft_layers[n][0] is weights whilst 1 is baises
        eq_system.subst(weight_symbol, fft_layers[i][0])
        eq_system.subst(bias_symbol, fft_layers[i][1])
    
    return eq_system


def model_create_equation(model_dir):
    # create prequesties
    model = keras.models.load_model(model_dir)
    peq_system = get_poly_eqs(model.layers.len())
    
    # calculate fft
    from scipy import fft as fft
    layers = []
    for layer in model.layers:
        # fft for interpolation or finding the polynomial
        # fft == inverse laplace.
        w_layer = layer.get_weights()[0]
        b_layer = layer.get_weights()[1]
        
        layer = (w_layer, b_layer)
        fft_layers.append(fft_layer)
    
    # calculate subsitute into system
    # use units of the power circle?
    #peq_system = subst_into_system(fft_layers, peq_system)



class Ideas(Scene):
    def construct(self):
        # intro
        title = Tex(r"A mathematical journey into Deep neural networks")
        prequesites = Tex(r"Prequesite ideas")
        VGroup(title,prequesites)

        self.play(Write(title))
        self.wait()
        self.play(FadeIn(prequesites), FadeOut(title))
        self.wait()

