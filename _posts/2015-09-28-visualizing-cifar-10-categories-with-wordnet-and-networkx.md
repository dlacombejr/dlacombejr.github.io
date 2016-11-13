---
layout: post
title:  "Visualizing CIFAR-10 Categories with WordNet and NetworkX"
date:   2015-09-28 
category: programming
tags: []
---

In this post, I will describe how the object categories from [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) can be visualized as a [semantic network](https://en.wikipedia.org/wiki/Semantic_network). CIFAR-10 is a database of images that is used by the computer vision community to benchmark the performance of different learning algorithms. For some of the work that I'm currently working on, I was interested in the semantic relations between the object categories, as other research has been in the past. We can do this by defining their relations with [WordNet](https://wordnet.princeton.edu/) and then visualizing them using [NetworkX](https://networkx.github.io/) combined with [Graphviz](http://www.graphviz.org/). 

<!--more-->

### Python Dependencies

Before being able to run the code described in this post, there are a couple of dependencies that must be installed (if not already on your machine). This includes the [NetworkX installation](https://networkx.github.io/documentation/latest/install.html), [NLTK installation](http://www.nltk.org/install.html), and [Graphviz installation](http://www.graphviz.org/Download..php). Also, after installing NLTK, `import nltk` and use `nltk.download()` to futher install the `wordnet` and `wordnet_ic` databases. You should be all set at this point!

## Visualizing CIFAR-10 Semantic Network

For this code demonstration, we do not actually need the CIFAR-10 dataset, but rather its object categories. One alternative would be to download the dataset and use the `batches.meta` file to import the labels. For simplicity, I instead just list out the categories and put them into a set. 

{% highlight python linenos %}
categories = set()
categories.add('airplane')
categories.add('automobile')
categories.add('bird')
categories.add('cat')
categories.add('deer')
categories.add('dog')
categories.add('frog')
categories.add('horse')
categories.add('ship')
categories.add('truck')
{% endhighlight %}

Now we need to define a function that, beginning with a given object class, recursively adds a node and an edge between it and its [*hypernym*](https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy) all the way up to the highest node (i.e., "entity"). I found [this post](http://www.randomhacks.net/2009/12/29/visualizing-wordnet-relationships-as-graphs/) that demonstrated code that could do this, so I borrowed it and modified it for my purposes. The major addition was to extend the graph building function to mulitple object categories. We define a function `wordnet_graph` that builds us our network:

{% highlight python linenos %}
import networkx as nx
import matplotlib.pyplot as pl
from nltk.corpus import wordnet as wn

def wordnet_graph(words):
    
    """
    Construct a semantic graph and labels for a set of object categories using 
    WordNet and NetworkX. 
    
    Parameters: 
    ----------
    words : set
        Set of words for all the categories. 
        
    Returns: 
    -------
    graph : graph
        Graph object containing edges and nodes for the network. 
    labels : dict
        Dictionary of all synset labels. 
    """
    
    graph = nx.Graph()
    labels = {}
    seen = set()
    
    def recurse(s):
        
        """ Recursively move up semantic hierarchy and add nodes / edges """  

        if not s in seen:                               # if not seen...
            seen.add(s)                                 # add to seen
            graph.add_node(s.name)                      # add node
            labels[s.name] = s.name().split(".")[0]     # add label
            hypernyms = s.hypernyms()                   # get hypernyms

            for s1 in hypernyms:                        # for hypernyms
                graph.add_node(s1.name)                 # add node
                graph.add_edge(s.name, s1.name)         # add edge between
                recurse(s1)                             # do so until top
     
    # build network containing all categories          
    for word in words:                                  # for all categories
        s = wn.synset(str(word) + str('.n.01'))         # create synset            
        recurse(s)                                      # call recurse
    
    # return the graph and labels    
    return graph , labels   
{% endhighlight %}

Now we're ready to create the graph for visualizing the semantic network for CIFAR-10. 

{% highlight python linenos %}
# create the graph and labels
graph, labels = wordnet_graph(categories)

# draw the graph
nx.draw_graphviz(graph)
pos=nx.graphviz_layout(graph)
nx.draw_networkx_labels(graph, pos=pos, labels=labels)
pl.show()  
{% endhighlight %}

The resulting semantic network should look like the following:

{% include image.html img="/assets/CIFAR_10-wordnet.png" title="CIFAR_10-wordnet" caption="Semantic Network for CIFAR-10" %}

We can see that from *entity*, the main branch between categories in CIFAR-10 is between artifacts and living things. The object categories themselves tend to be terminal nodes (except for dog). 

## Quantifying Semantic Similarity

We can also use WordNet to quantify the semantic distance between two given object categories. Developing quantifications for semantic similarity is an area of ongoing study and the NLTK includes a couple variations. Here, we use a simple `path_similarity` quantification which is the length of the shortest path between two nodes, but many others can be implemented by using the `wordnet_ic` dataset and defining an information content dictionary (see [here](http://www.nltk.org/howto/wordnet.html)). 

To find the semantic distance between all object categories, we create an empty similarity matrix of size $$N \times N$$, where $$N$$ equals the number of object categoes, and iteratively calculate the semantic similarity for all pair-wise comparisons. 

{% highlight python linenos %}
import numpy as np
from nltk.corpus import wordnet_ic

# empty similarity matix
N = len(categories)
similarity_matrix = np.zeros((N, N))

# initialize counters
x_index = 0
y_index = 0
# loop over all pairwise comparisons
for category_x in categories:
    for category_y in categories:
        x = wn.synset(str(category_x) + str('.n.01')) 
        y = wn.synset(str(category_y) + str('.n.01')) 
        # enter similarity value into the matrix
        similarity_matrix[x_index, y_index] = x.path_similarity(y) 
        # iterate x counter
        x_index += 1
    # reinitialize x counter and iterate y counter   
    x_index = 0
    y_index += 1

# convert the main diagonal of the matrix to zeros       
similarity_matrix = similarity_matrix * abs(np.eye(10) - 1)
{% endhighlight %}

We can then visualize this matrix using Pylab. I found [this notebook](http://nbviewer.ipython.org/gist/joelotz/5427209) that contained some code for generating a nice comparison matrix. I borrowed that code and only made slight modifications for the current purposes. This code is as follows:

{% highlight python linenos %}
# Plot it out
fig, ax = pl.subplots()
heatmap = ax.pcolor(similarity_matrix, cmap=pl.cm.Blues, alpha=0.8)

# Format
fig = pl.gcf()
fig.set_size_inches(8, 11)

# turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(similarity_matrix.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(similarity_matrix.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

# Set the labels

# label source:https://en.wikipedia.org/wiki/Basketball_statistics
labels = []
for category in categories:
    labels.append(category)


# note I could have used nba_sort.columns but made "labels" instead
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels(labels, minor=False)

# rotate the x-axis labels
pl.xticks(rotation=90)

ax.grid(False)

# Turn off all the ticks
ax = pl.gca()
ax.set_aspect('equal', adjustable='box')

for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
{% endhighlight %}

This generates the following visualization of the semantic similiary matrix for the CIFAR-10 object categories:

{% include image.html img="/assets/semantic_similarity_matrix.png" title="semantic_similarity_matrix" caption="Semantic Similarity Matrix for CIFAR-10 Categories" %}

In this image, bluer colors represent higher similarity (neglecting the main diagonal which was forced to zero for better visualization). As is apparent, all of the object categories belonging to either the artifact or living_thing major branches are closely similar to one another and very different from objects in the opposite branch.  Now these semantic distances between object categories can be used for many other types of analyses.
