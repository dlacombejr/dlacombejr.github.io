---
layout: post
title:  "Topographic Locally Competitive Algorithm"
date:   2015-09-13 17:02:29
category: programming
tags: [sparse-coding, LCA, MATLAB]
---

Recent studies have shown that, in addition to the emergence of receptive fields similar to those observed in biological vision using [sparse representation models](http://dlacombejr.github.io/programming/2015/09/15/sparse-filtering-implemenation-in-theano.html), the self organization of said receptive fields can emerge from group sparsity constraints. Here I will briefly review research demonstrating topological organization of receptive fields using group sparsity principles and then describe a two-layer model implemented in a Locally Comptetitive Algorithm that will be termed Topographical Locally Competitive Algorithm (tLCA). 

<!--more-->


## Topographic Organization of Receptive Fields

In biological vision, receptive fields in early visual cortex are organized into orientation columns where adjacent columns have selectivity close in feature space. The global appearance of selectivity to oreintation across the coritical sheet is that of smooth transitions between orientation preference of columns and the classic *pinwheel* features where orientation column selectivities meet at a *singularity* (see image below). 

<!-- {:refdef: style="text-align: center;"}
![placeholder](/assets/orientation_columns.png "Orientation Dominance Columns")
{: refdef} -->

{% include image.html img="/assets/orientation_columns.png" title="Orientation Dominance Columns" caption="Orientation dominance columns across the cortical surface" url="http://www.ib.cnea.gov.ar/~redneu/2013/BOOKS/Principles%20of%20Neural%20Science%20-%20Kandel/gateway.ut.ovid.com/gw2/ovidweb.cgisidnjhkoalgmeho00dbookimagebookdb_7c_2fc~33.htm"%}

A large amount of computational research has explored the mechanisms underlying such organization {% cite swindale1996development %}. More recent research has learned the self-organization of feature detectors based on the natural statistics of images when structured sparsity is imposed {% cite hyvarinen2001topographic jia2010self kavukcuoglu2009learning welling2002learning %}.[^fn-biological_accuracy] Most of these models involve a two layers where the activations of the first layer are square rectified and projected up to a second layer based on locally defined connections. If we have activations $$a^{(1)}$$ in the first layer given by:

\begin{equation}
a^{(1)} = \mathbf{w}^T \mathbf{x}
\end{equation}

where $$\mathbf{w}$$ is the weight matrix and $$\mathbf{x}$$ is the input data, these can then be projected up to a second layer unit $$a_i^{(2)}$$ given the local connections defined by $$k$$ overlapping neighborhoods $$H$$:

\begin{equation}
a_i^{(2)} = \sqrt{\sum_{j \in H_i} (a^{(1)}_j)^2	}
\end{equation}

Thus, the activation of each unit in the second level is the sum of sqares of adjacent units in the first layer as defined by a local connectivity matrix that can either be binary or have some distribution across the nieghborhood (below is an example of 3 X 3 overlapping neighborhoods). 

<!-- {:refdef: style="text-align: center;"}
![](/assets/grouping2.png)
{: refdef}
 -->
{% include image.html img="/assets/grouping2.png" title="Grouping" caption="An example showing 3x3 overlapping neighborhoods" %}

To avoid edge artifacts, these neighborhoods are also commonly defined to be toroidal so that each unit in a given layer has an equal number of neighbors. 

<!-- {:refdef: style="text-align: center;"}
![](/assets/Torus_from_rectangle.gif)
{: refdef}
 -->
{% include image.html img="/assets/Torus_from_rectangle.gif" title="torus" caption="Demonstration of converting 2d plane to torus" url="https://en.wikipedia.org/wiki/File:Torus_from_rectangle.gif"%}


Thus the optimization objective for a *sparse-penalized least-squares reconstruction* model with the aforementioned architecture would be:

\begin{equation}
\min_{\mathbf{\alpha}\in\mathbb{R}^m}	\vert\vert \mathbf{x} - \mathbf{w}^T a^{(1)} \vert\vert^2_2+\lambda\vert\vert 	a^{(1)}	+a^{(2)}  \vert\vert_1
\end{equation}

where, as before, $$\lambda$$ is a sparsity tradeoff parameter. 



## Self-Organization of Receptive Fields using Locally Competitive Algorithms

### Locally Competitive Algorithm

Locally competitive algorithms {% cite rozell2007locally %} are dynamic models that are implementable in hardware and converge to good solutions for sparse approximation.  In these models, each unit has an state $$u_m(t)$$, and when presented with a stimulus $$s(t)$$, each unit begins accumulating activity that leaks out over time (much like a bucket with small holes on the bottom). When units reach a threshold $$\lambda$$, they begin exerting inhibition over their competitors weighted by some function based on similarity or proximity in space. The states of a given unit $$m$$ is represented by the nonlinear ordinary differential equaion

$$\dot{u}_m(t)=\frac{1}{\tau}\bigg[b_m(t)-u_m(t)-\sum_{n\neq m}G_{m,n}a_n(t)\bigg] $$

where $$b_m(t)=\langle\phi_m,{x}(t)\rangle$$ represents increased activation proportional to the receptive field's similarity to the incoming input. The internal states of each unit and thus the degree of inhibition that they can exert are expressed as

$$
 a_m=T_m(u)   =\left\{ 
\begin{array}{ccl} 
0, &u\leq \lambda \\ \\
u, &u > \lambda \\
\end{array}
\right. 
$$

which simply means that if the state of a unit is below the threshold, its internal state is zero, and if the state is above threshold, it's internal state is a linear function of $$u$$. This inhibition is finally wieghted based on the similarity between two units $$G_{m,n}=\langle\phi_m,\phi_n\rangle$$ ensuring that redundant feature representations are not used for any given input and a sparse approximation is achieved. 

### tLCA Model

Here I will introduce a two-layer Locally Competitive Algorithm that I will call Topographical Locally Competitive Algorithm (tLCA). The general procedure is to first determine the initial activity of the first layer, immediately project it to the second layer in a fast feedforward manner, perform LCA at the second layer, project the activity back down to the first layer, and then perform LCA on the first layer (see figure for schematic illustration).

<!-- {:refdef: style="text-align: center;"}
![](/assets/tLCA.png)
{: refdef}
 -->

{% include image_long.html img="/assets/tLCA.png" title="network diagram" caption="Illustration of the network architecture and procedure. The activation for the first layer (blue) is calculated and then local connections from the second layer (orange) to the first allow for it to pool over a neighborhood of units (cyan). Local competition is then performed on the second layer. In the right panel, after LCA on the second layer terminates, activations are then projected back down to the first layer via local connections. Finally LCA on the first layer is conducted until termination. " %}

Now I will walk through the steps of bulding the model (to see all code navigate to my [tLCA] repository). To begin building the model, we will first define some parameters: 

{% highlight matlab linenos %}
% set environment parameters
neurons = 121;      				% number of neurons
patch_size = 256;   				% patch size
batch_size = 1000;  				% batch size
thresh = 0.1;       				% LCA threshold 
h = .005;           				% learning rate
blocksize = 3;     				% neighborhood size
maxIter = 1000;    				% maximum number of iterations
{% endhighlight %}

I can then randomly initialize the wieghts of the network and constrain them to lie on the $$\ell_2$$ ball via normalization:

{% highlight matlab linenos %}
W = randn(patch_size, neurons); 		% randomly initialize wieghts
W = W * diag(1 ./ sqrt(sum(W .^ 2, 1)));	% normalize the weights
{% endhighlight %}

Next we need to define the local connectivities between the first layer and the second layer. These weights are held constant and are not trained like the weights of the first layer that connect to the input. To do so, we define a function `gridGenerator` with arguments `neurons` and `filterSize` and returns a `group x neurons` matrix `blockMaster` that contains binary row vectors with filled entries corresponding to neurons that belong to the i<sup>th</sup> group. 

{% highlight matlab linenos %}
function blockMaster = gridGenerator(neurons, filterSize)

% determine grid dimensions
gridSize = sqrt(neurons);

% create matrix with grids
blockMaster = zeros(neurons, neurons);
c = 1;
x = zeros(gridSize, gridSize);
x(end - (filterSize - 1):end, end - (filterSize - 1):end) = 1;
x = circshift(x, [1, 1]);
for i = 1:gridSize 
    for j = 1:gridSize 
        temp = circshift(x, [i, j]);
        blockMaster(c,:) = temp(:)';
        c = c + 1; 
    end
end
{% endhighlight %}

This works by first creating a binary matrix with ones over the group centered at `x(1,1)` (lines 9-11); because it is toriodal, there are ones on opposite sides of the matrix. Then, for all groups, it shifts this primary matrix around until all group local connections have been created and saved into the master matrix. 

Now that we have a means of projecting the first layer activation up to the second layer, we need to define how inhibition between units in the second layer should be weighted. We can define the mutual inhibiiton between two units in the second layer as being proportional to how many units in the first layer share their local connections. This can be conveniently created as follows:

{% highlight matlab linenos %}
% create group inhibition weight matrix
G2 = blockMaster * blockMaster'; 
G2 = G2 ./ max(max(G2)); 
G2 = G2 - eye(neurons); 
{% endhighlight %}

Lastly we need to also set up a similarity matrix for all pairwise connections between units. In the traditional LCA, this was computed as the similarity between receptive fields as described previously. Here we instead compute similarity as Euclidean distance in simulated cortical space. We can compute the distance of each unit to all other units using the function `lateral_connection_generator`:

{% highlight matlab linenos %}
function master = lateral_connection_generator(neurons)

% define grid size
dim = sqrt(neurons);

% create list of all pairwise x-y coordinates 
x = zeros(dim * dim, 2); 
c = 1; 
for i = 1:dim
    for j = 1:dim
        x(c, :) = [i, j]; 
        c = c + 1; 
    end
end

% create distance matrix of each cell from the center of the matrix
center_index = ceil(neurons / 2);
center = x(center_index, :); 
temp = zeros(dim, 1); 
for j = 1:size(x, 1)
    temp(j) = norm(center - x(j, :)); 
end
temp = reshape(temp, [dim, dim]); 

% shift the center of the matrix (zero distance) to the bottom right corner
temp = circshift(temp, [center_index - 1, center_index - 1]); 

% create master matrix 
master = zeros(neurons, neurons);
c = 1; 
for i = 1:dim
    for j = 1:dim
        new = circshift(temp, [j, i]); 
        master(c, :) = new(:)'; 
        c = c + 1; 
    end
end
{% endhighlight %}

Now we are ready to actually run the neural network and analyze its characteristics. Image patches that were preselected from natural images and preprocessed through normalization are read in and assigned to the variable `X`. Then we loop through each training iteration and perform the following procedure:

* Normalize the weights as a form of regularization (as done previously)

{% highlight matlab linenos %}
W = W * diag(1 ./ sqrt(sum(W .^ 2, 1)));		% normalize the weights
{% endhighlight %}

* Rapidly feed forward the activation through first and on to the second level

{% highlight matlab linenos %}
b1 = W' * X; 				   	% [neurons X examples]
b2 = (blockMaster * sqrt(b1 .^ 2)) / blocksize; 	% [groups X examples]
{% endhighlight %}

* Perform LCA at layer 2

{% highlight matlab linenos %}
u2 = zeros(groups,batch_size);
for i = 1:5
    a2 = u2 .* (abs(u2) > thresh);
    u2 = 0.9 * u2 + 0.01 * (b2 - G2 * a2);        
end
a2=u2.*(abs(u2) > thresh); 				% [groups, batch_size]
{% endhighlight %}

* Project the activations back down to the first layer

{% highlight matlab linenos %}
a1 = blockMaster' * a2; 				% [neurons X batch_size]
a1 = a1 .* b1; 					% weight by first level activation
{% endhighlight %}

* Perform LCA on the first layer

{% highlight matlab linenos %}
u1 = a1;
for l =1:10
    a1=u1.*(abs(u1) > thresh);
    u1 = 0.9 * u1 + 0.01 * (b1 - G1*a1);
end
a1=u1.*(abs(u1) > thresh); 				% [groups, batch_size]
{% endhighlight %}

* Update the weights

{% highlight matlab linenos %}
W = W + h * ((X - W * a1) * a1');
{% endhighlight %}

Running this code using `maxIter` as set above takes just over a minute. The features that are learned replicate those found in the literature and they also self organize as has been found in the studies cited. An important observation is that the receptive fields organize by both orientation *and* spatial frequency, whereas lateral connections alone (run `latLCA.m` for comparison) only leads to some organization of orientation. Therefore, performing LCA in a two-layer network as we did here seems to be necessary to get good self organization along both dimensions. It is also important to note that phase appears to organize randomly, and this is due to the square rectification of the first layer (i.e., a counter-phase stimulus may result in a negative activation, but this will be rectified into a positive activation). 

{:refdef: style="text-align: center;"}
![](/assets/tLCA_weights.png)
{: refdef}



[^fn-biological_accuracy]: Although these models have the advantage of being driven by natural image statistics, they also suffer from some biological implausibility {% cite antolik2011development %}.

<center><b>References</b></center>
{% bibliography --cited %}
----------------