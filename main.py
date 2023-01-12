from manim import (Tex, Scene, 
                    VGroup, Write,
                    FadeIn, FadeOut)

"""
script start

General Outline
Nerual nets are function approximators
We can extract the function the neural net is approximating

Math statements

A matrix is transformation of a vector
A matrix of directions transforming a vector is a function
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
if we take the idea of integration, taking the input space of vector V

prop:
we apply the same matrix of weights (function on the basis) to a new starting point(training batches),
and widen the search net of the training samples until the whole input space is covered
this is additive fourier transforms of trained weights


ideas:
infinite weights => function local minima
matrix exponents
divergence and curl for functions => Discrete matrix transforms
borsuk ulam theorem?
e^Mt = transform / rotation
meta-networks => deriving the function from networks and then inverse fourier 
to create genralized function. Same idea as finding the transform for training batches.

eigenstates for input matricies
foruier transforms => to create human readable functions
quanta + tunneling to find absoloute minima =>

minibatch selectors => HYP: 
taking the largest distance inputs then getting smaller distances 
until finding convergence 
Once convergence is found for local minima i.e. det == 0 

find if convergable first?

"""

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

