---
layout: post
title:  "Sparse Filtering in Theano"
date:   2015-09-13 17:02:29
category: programming
tags: [sparse-filtering, sparse-coding, theano, python]
---

Sparse Filtering is a form of unsupervised feature learning that learns a sparse representation of the input data without directly modeling it. This algorithm is attractive because it is essentially hyperparameter-free making it much easier to implement relative to other existing algorithms, such as Restricted Boltzman Machines, which have a large number of them. Here I will review a selection of sparse representation models for computer vision as well as Sparse Filtering's place within that space and then demonstrate an implementation of the algorithm in Theano combined with the L-BFGS method in SciPy's optimizaiton library. 

<!--more-->

## Sparse Representation Models for Computer Vision

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

The second step is to normalize across the columns, which again maps the entries to the unit ball. This makes the rows about equally active,  introducing competition between the features and thus removing the need for an orthogonal basis. Sparse filtering prevents degenerate situations in which the same features are always active.\cite{ngiam2011sparse}   

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



## Limitations

<center><b>References</b></center>
{% bibliography --cited %}