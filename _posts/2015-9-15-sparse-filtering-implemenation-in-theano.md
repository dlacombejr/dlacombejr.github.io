---
layout: post
title:  "Sparse Filtering in Theano"
date:   2015-09-13 
category: programming
tags: [sparse-filtering, sparse-coding, theano, python]
---

Sparse Filtering is a form of unsupervised feature learning that learns a sparse representation of the input data without directly modeling it. This algorithm is attractive because it is essentially hyperparameter-free making it much easier to implement relative to other existing algorithms, such as Restricted Boltzman Machines, which have a large number of them. Here I will review a selection of sparse representation models for computer vision as well as Sparse Filtering's place within that space and then demonstrate an implementation of the algorithm in Theano combined with the L-BFGS method in SciPy's optimizaiton library. 

<!--more-->

## Sparse Representation Models for Computer Vision ##

Models employing sparsity-inducing norms are ubiquitous in the statistical modeling of images. Their employment is strongly motivated by the inextricably woven web of a sparse code's efficiency, functionality, and input distribution match --- a rather uncanny alignment of properties. Representing an input signal with a few number of active units has obvious benefits in efficient energy usage; the fewer units that can be used to provide a good representation, without breaking the system, the better. A sparse coding scheme also has logically demonstrable functional advantages over the other possible types (i.e., local and dense codes) in that it has high representational and memory capacites (representatoinal capacity grows exponentially with average activity ratio and short codes do not occupy much memory), fast learning (only a few units have to be updated), good fault tolerance (the failure or death of a unit is not entirely crippling), and controlled interference (many representations can be active simultaneously){% cite foldiak1995sparse %}. Finally, and perhaps most mysteriously, a sparse code is a good representational scheme because it matches the sparse structure, or non-Gaussianity of natural images {% cite simoncelli2001natural %}. That is, images can be represented as a combination of a sparse number of elements. Because a sparse code matches the sparse distribution of natural scenes, this provides a good statistiacal model of the input, which is useful because...

{% quote hyvarinen2009natural %}
...such models provide the prior probabilities needed in Bayesian inference, or, in general, the prior information that the visual system needs on the environment. These tasks include denoising and completion of missing data. So, sparse coding models are useful for the visual system simply because they provide a better statistical model of the input data.
{% endquote %}

### Sparse Coding

In the mid 90s, a seminal article by Olshausen and Field marked the beginning of a proliferation of research in theoretical neuroscience, computer vision, and machine learning more generally. There, they first introduced the computational model of sparse coding {% cite olshausen1996emergence %}  and demonstrated the ability to learn units with receptive fields strongly resembling those observed in biological vision when trained on natural images. Sparse coding is based on the assumption that an input image $${I(x, y)}$$ can be modeled as a linear combination of sparsely activated representational units $${\phi_i(x, y)}$$:

\begin{equation}
{I(x, y)} = \sum_i a_i \phi_i(x, y)
\end{equation}

Given this linear generative model of images, the goal of sparse coding is then to find some representational units $${\phi_i(x, y)}$$ that can be used to represent an image using a sparse activity coefficient vector $$a$$ (i.e., one that has a leptokurtotic distribution with a large peak around zero and heavy tails as can be seen in the figure below). 

![placeholder](/assets/sparse_gaussian_comp.png "Comparison Between Gaussian and non-Gaussian Distribution")

The optmization problem for finding such a sparse code can be formalized by minimizing the following cost function:

\begin{equation}
	E = - {\sum_{x, y} \bigg[ {I(x, y)} - \sum_i a_i \phi_i(x, y) \bigg] ^2} - {\sum_i }  S\Big(\frac{a_i} {\sigma}\Big)
\end{equation}

where $$S(x)$$ is some non-linear function that penalizes non-sparse activations and $$\sigma$$ is a scaling constant. We can see that this is basically a combination of a reconstruction error and a sparsity cost, what can be referred to as <em>sparse-penalized least-squares reconstruction</em> and can be generally represented by:

\begin{equation}
\text{cost = [reconstruction error] + [sparseness]}
\end{equation}

More generally, this form of problem falls under the more general class of sparse approximation where a good subset of a dictionary $$\mathbf{D}$$ must be found to reconstruct the data:

\begin{equation}
\min_{\mathbf{\alpha}\in\mathbb{R}^m}	\frac{1}{2}	\vert\vert \mathbf{x} - \mathbf{D}\alpha\vert\vert^2_2+\lambda\vert\vert \alpha\vert\vert_1
\end{equation}

However, in this case, $$\mathbf{D}$$ is not known and thus makes this an unsupervised learning problem. 

<!-- ### Independent Components Analysis

Sparse coding can be formalized probabalistically as independent component analysis (ICA), a statistical generative model that produces latent variables assumed to be independent. Consider a set of random variables, $$s_1, ..., s_n$$. We can define the independence between this set of variables formally as a <em>factorizable</em> joint pdf: 

\begin{equation}
p(s_1, ...,s_n) = \prod p_i(s_i)
\end{equation}

Thus, knowing any information about the values that a given variable $$s_i$$ gives us no predictive power in estimating the values of any other variable in the set. 

Optimal measure of sparsity is 

\begin{equation}
h_{opt}(s^2) = \text{log} \, p_s(s)
\end{equation} -->

<!-- 
### Universal Cortical Algorithm -->

## Sparse Filtering

Sparse Filtering {% cite ngiam2011sparse %} is an unsupervised learning technique that does not directly model the data (i.e., it has no reconstruction error term in the cost function). The goal of the algorithm is to learn a dictionary $$\mathbf{D}$$ that provides a sparse representation by minimizing the normalized entries in a feature value matrix. For each iteration of the algorithm: 

1. $$\ell_2$$ normalization across rows
2. $$\ell_2$$ normalization across columns
3. Objective function = $$\ell_1$$ norm of normalized entries

The remaining portion of this subsection is an excerpt from {% cite hahn2015deep %}:

Let $$\mathbf{F}$$ be the feature value matrix to be normalized, summed, and minimized. The components 

\begin{equation}
f^{(i)}_j
\end{equation}

represent the $$j^{\text{th}}$$ feature value ($$j^{\text{th}}$$ row) for the $$i^{\text{th}}$$ example ($$i^{\text{th}}$$ column), where 

\begin{equation}
f^{(i)}_j=\mathbf{w}_j^T\mathbf{x}^{(i)}
\end{equation}

Here, the $$\mathbf{x}^{(i)}$$ are the input patches and  $$\mathbf{W}$$ is the weight matrix. Initially random, the weight matrix is updated iteratively in order to minimize the Objective Function.

In the first step of the optimization scheme,

\begin{equation}
\widetilde{\mathbf{f}}_j=\frac{\mathbf{f}_j}{\vert\vert\mathbf{f}_j\vert\vert_2}
\end{equation}

Each feature row is treated as a vector, and mapped to the unit ball by dividing by its $$\ell_2$$-norm. This has the effect of giving each feature approximately the same variance. 

The second step is to normalize across the columns, which again maps the entries to the unit ball. This makes the rows about equally active,  introducing competition between the features and thus removing the need for an orthogonal basis. Sparse filtering prevents degenerate situations in which the same features are always active {% cite ngiam2011sparse %}. 

\begin{equation}
\hat{\mathbf{f}}^{(i)}=\frac{\widetilde{\mathbf{f}}^{(i)}}{\vert\vert\widetilde{\mathbf{f}}^{(i)}\vert\vert_2}
\end{equation}

The normalized features are optimized for sparseness by minimizing the $$\ell_1$$ norm. That is, minimize the Objective Function, the sum of the absolute values of all the entries of $$\mathbf{F}$$. For datasets of $$M$$ examples we have the sparse filtering objective:

<!-- \begin{equation}
 -->

 $$\text{minimize}\quad \sum_{i=1}^M \left\vert\left\vert \hat{\mathbf{f}}^{(i)}\right\vert\right\vert_1= \sum_{i=1}^M \left\vert\left\vert \frac{\widetilde{\mathbf{f}}^{(i)}}{\vert\vert\widetilde{\mathbf{f}}^{(i)}\vert\vert_2}\right\vert\right\vert_1$$ {: .center}
<!-- \end{equation}
 -->

The sparse filtering objective is minimized using a Limited-memory Broyden--Fletcher--Goldfarb--Shanno (L-BFGS) algorithm, a common iterative method for solving unconstrained nonlinear optimization problems.


## Implementation in Theano

Theano is a powerful Python library that allows the user to define and optimize functions that are compiled to machine code for faster run time performance. One of the niceset features of this package is that it performs automatic symbolic differentation. This means we can simply define a model and its cost function and Theano will calculate the gradients for us! This frees the user from analytically deriving the gradients and allows us to explore many different model-cost combinations much more quickly. However, one of the drawbacks of this library is that it does not come prepackaged with more sophisticated optimization algorithms, like L-BFGS. Other Python libraries, such as SciPy's optimize library do contain these optimization algorithms and here I will show how they can be integrated with Theano to optimize sparse filters with respect to their cost function described above. 

First we define a SparseFiter class which performs the normalization scheme formalized above. 

{% highlight python linenos %}
import theano
from theano import tensor as t

class SparseFilter(object):

    """ Sparse Filtering """

    def __init__(self, w, x):

        """
        Build a sparse filtering model.

        Parameters:
        ----------
        w : ndarray
            Weight matrix randomly initialized.
        x : ndarray (symbolic Theano variable)
            Data for model.
        """

        # assign inputs to sparse filter
        self.w = w
        self.x = x

    def feed_forward(self):

        """ Performs sparse filtering normalization procedure """

        f = t.dot(self.w, self.x.T)               # initial activation values
        fs = t.sqrt(f ** 2 + 1e-8)              # numerical stability
        l2fs = t.sqrt(t.sum(fs ** 2, axis=1))   # l2 norm of row
        nfs = fs / l2fs.dimshuffle(0, 'x')      # normalize rows
        l2fn = t.sqrt(t.sum(nfs ** 2, axis=0))  # l2 norm of column
        f_hat = nfs / l2fn.dimshuffle('x', 0)   # normalize columns

        return f_hat

    def get_cost_grads(self):

        """ Returns the cost and flattened gradients for the layer """

        cost = t.sum(t.abs_(self.feed_forward()))
        grads = t.grad(cost=cost, wrt=self.w).flatten()

        return cost, grads
{% endhighlight %}

When this object is called, it is initialized with the passed weights and data variables. It also has a `feed_forward` method for getting the normalized activation values for $$\mathbf{F}$$ as well as a `get_cost_grads` method that returns the cost (defined above) and the gradients wrt the cost. Note that in this implementation, the gradients are flattened out; this has to do with making Theano compatible with SciPy's optimization library as will be described next. 

Now we need to define a function that, when called, will compile a Theano training function for the `SparseFilter` based on it's cost and gradients at each training step as well as a callable function for SciPy's optimization procedure that does the following steps:

1. Reshape the new weights `theta_value` consistent with how they are initialized in the model and convert to float32
2. Assign those reshaped and converted weights to the model's weights
3. Get the cost and the gradients based on the compiled training function
4. Convert the weights back to float64 and return

Note that in step #3, the gradients returned are already vectorized based on the `get_cost_grads` method of the `SparseFilter` class for compatability with SciPy's optimization framework. The code for accomplishing this is as follows: 

{% highlight python linenos %}
import numpy as np

def training_functions(data, model, weight_dims):

    """
    Construct training functions for the model.

    Parameters:
    ----------
    data : ndarray
        Training data for unsupervised feature learning.

    Returns:
    -------
    train_fn : list
        Callable training function for L-BFGS.
    """

    # compile the Theano training function
    cost, grads = model.get_cost_grads()
    fn = theano.function(inputs=[], outputs=[cost, grads],
                         givens={model.x: data}, allow_input_downcast=True)

    def train_fn(theta_value):

        """
        Creates a shell around training function for L-BFGS optimization
        algorithm such that weights are reshaped before calling Theano
        training function and outputs of Theano training function are
        converted to float64 for SciPy optimization procedure.

        Parameters:
        ----------
        theta_value : ndarray
            Output of SciPy optimization procedure (vectorized).

        Returns:
        -------
        c : float64
            The cost value for the model at a given iteration.
        g : float64
            The vectorized gradients of all weights
        """

        # reshape the theta value for Theano and convert to float32
        theta_value = np.asarray(theta_value.reshape(weight_dims[0],
                                                     weight_dims[1]),
                                 dtype=theano.config.floatX)

        # assign the theta value to weights
        model.w.set_value(theta_value, borrow=True)

        # get the cost and vectorized grads
        c, g = fn()

        # convert values to float64 for SciPy
        c = np.asarray(c, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)

        return c, g

    return train_fn
{% endhighlight %}

Now that we have the model defined and the training environment, we can build the model and visualize what it learns. First we read in some data and preprocess it by centering the mean at zero and whitening to remove pairwise correlations. Finally we convert the data to float32 for GPU compatability. 

{% highlight python linenos %}
from scipy.io import loadmat
from scipy.cluster.vq import whiten

data = loadmat("patches.mat")['X'] 		# load in the data
data -= data.mean(axis=0)			# center data at mean
data = whiten(data)				# whiten the data
data = np.float32(data.T)			# convert to float32
{% endhighlight %}

Next we define the model variables, including the network architecture (i.e., number of neurons and their weights), the initial weights themselves, and a symbolic variable for the data. 

{% highlight python linenos %}
from init import init_weights

weight_dims = (100, 256)       		# network architecture
w = init_weights(weight_dims)   		# random weights
x = t.fmatrix()                 		# symbolic variable for data
{% endhighlight %}

The imported method `init_weights` simply generates random weights with zero mean and unit variance. In addition, these weights are deemed "shared" variables so that they can be updated across all function that they appear in and are designated as float32 for GPU compatability. With this in place, we can then build the Sparse Filtering model and the training functions for its optimization. 

{% highlight python linenos %}
model = SparseFilter(w, x)
train_fn = training_functions(data, model, weight_dims)
{% endhighlight %}

Finally, we can train the model using SciPy's optimization library. 

{% highlight python linenos %}
from scipy.optimize import minimize

weights = minimize(train_fn, model.w.eval().flatten(),
                   method='L-BFGS-B', jac=True,
                   options={'maxiter': 100, 'disp': True})
{% endhighlight %}

With the maximum number of iterations set at 100, this algorithm converges well under a minute. We can then visualize the representations that it has learned by grabbing the final weights and reshaping them. 

{% highlight python linenos %}
import visualize

weights = weights.x.reshape(weight_dims[0], weight_dims[1])
visualize.drawplots(weights.T, 'y', 'n', 1)
{% endhighlight %}

{% include image.html img="/assets/sf_weights.png" title="sf_weights" caption="Sparse Filtering Weights" %}

As we can see, Sparse Filtering learns edge-like feature detectors even withough modeling the data directly. Similar outcomes can also be acquired using standard gradient descent methods. 

## References

<!-- ## Limitations
 -->
<!-- <center><b>References</b></center>
 -->{% bibliography --cited %}