# Silver-Computing-Machine 
A model, dataset agnostic method of mathematically analyzing the curvature of 
a neural network given it's weights and a test dataset, using Sheaf theory 
and Cech Cohomology - the computation variant of De Rhams Chomology which allows 
to find global maxims and minimums within a closed dataset. 

The high dimensional sheafified output of curvature is tested by using 
b spline interpolation onto the input source which gives 
N^2 results of the batch these images are then merged using a merge Mertens 
algorithm and retrained on a model with it's weights been reset.

This is then tested with the original models as a control, 
and compared using a T-test.

Test results may be found in test_output.json.
