# Graph Signal Samplings

## Why
Graphs can capture spacial relationships between things:
 - 3D Models (vertex mesh)
 - Network topology
 - Image: Neighbouring pixels as a graph lattice

Graph Signal Processing Examples:
 - Citation network analysis: Edges between vertices are citations between papers. Can cluster network of papers into groups.
 - Computer Vision: Correspondence between points. Can also use it for similarity tests.

Graphs represent manifolds:
 - A manifold describes a local surface on a geometric object
 - If you only look at the local neighbourhood then the surface behaves
   in a euclidean manner.

Graph Signals:
 - Assume that a graph is connected and undirected
 - A vertex signal is $g(i) : V \to R$
 - Graph Signal: A vector of all the possible vertex signals $g = [g(1), ..., g(n)]$
 - Example: Petersen graph. Height of a blue bar coming out from a node is the
   value of the signal.


Graph Fourier Transform:
 - Classic Fourier Transform: If you have a function that's defined on the real line
   you can do the fourier transform to expand this function in terms of complex
   exponentials. You can analyse how much high-or-low frequency parts you have
   in this function.
 - Transform: $\hat f(\epsilon) = \int_R f(t) e^{-2\pi i \epsilon t} dt$ (time to
   frequency domain, where $\epsilon$ is a frequency).
 - Inverse: $f(t) = \int_R \hat f (t) e^{2 \pi i \epsilon t} d \epsilon$

What does this have to do with graphs? Consider example of a ring
graph. Just a graph where each node has degree two. We can represent this
using weight matrix like:

$$
\begin{bmatrix}
  0 && 1 && 0 && ... && 1 \\
  1 && 0 && 1 && ... && 0 \\
  0 && 1 && 0 && ... && 0 \\
  0 && 0 && 1 && ... && 0 \\
  && && ... && && \\
  1 && ... && 1 && 0
\end{bmatrix}
$$

Then we also have a diagonal matrix with value $2$ along the diagonal representing
the degree of each edge.

We can create something called a *graph laplacian* matrix $L$, which is defined
as $D - W$.

If you take the eigenvectors of $L$, they become exactly the exponentials
used in the Fourier transform ($e^{-2 \pi \i \epsilon \t$). So you can represent
the classical Fouirer transform on a signal that is represented as a circular graph.

What about any type of graph?

$L = D - W$. You have a set of eigenvalues and eigenvectors, where the smallest
eigenvalue is zero. and corresponding eigenvectors.

The Graph Fourier Transform $g$ is defined on the vertices of the graph. Instead
of the complex exponentials you use the eigenvectrs of the graph laplacian to
great a function $\hat g$ which describes the frequency of the graph.

$\hat g(\lambda_I) = \sum^N_{i = 1} g(i) u(i)^{*^{(l)}}$

Also has an inverse:

$g(i) = \sum^{N - 1}_{i = 1} \hat g(\lambda_I) u(i)^{*^{(l)}}$

## What does the Frequency even mean?

In classical signal processing the interpretation of frequency is the rate
at which a component of the function is oscilating.

In graph signal processing we also have an interpretation.

(1) First eigenvalue: Always zero (corresponds to the DC component). This is
    the part of the signal that's always constant and not fluctuating.

(2) Other eigenvalues: For the eigenvector $u_i$ that is associated with low frequencies
    $\lambda_l < t$, the components captured by this eigenvector vary very slowly.

    Likewise for the eigenvector associated with high frequencies $lambda_l > t$, the
    components captured by this eigenvector vary very rapidly.

For two connected vertices $v_i, v_j$ and an eigenpair $\{u, \lambda\}$,
$u(i) is similar to $u(j)$ (two components of the eigenvector in those
slots $i$ and $j$) if $\lambda$ is low, but different if $\lambda$ is high.

For instance, we can see that the first eigenvector will only have positive values -
it does not give any additional subtractive power. If you look at the first eigenvector,
we can see that a lot of values on one side of that component are negative and values
on the other side are positive. If you look at the last eigenvector, you have very
rapid changes in terms of whether a graph signal value is positive or negative.

The frequency is the number of zero crossings in the graph - for each connected
vertex, if you look at the value of the eigenvalue at the $i$-th vertex and the
$j$-th vertex and their multiple is less than zero then you crossed zero.

## Domains
A graph signal can be represented in either the vertex doamin or the graph spectral domain:

Vertex domain: For instance an image, the signal is the grayscale value of each pixel

Spectral domain: How much do you have sharp discontinuities in the image? If you want
do to convolutions, you can define some kind of vector that represents a convolution
directly in the graph spectral domain and then put it back into the graph spectral domain.

## Operators

### Translation $f(t - 3)$
How do you select the "third previous vertex", since vertices are arbitrarily ordered. Not
well defined.

### Modulation? $e^k f(t)$
Translation in the graph spectral domain? Works in classical signal processing because the
frequency spectrum is continuous, but this doesn't work in graph signal processing since the
eigenspectrum is not continous.

### Downsampling?
What is "every other vertex"? How do you capture the structural properties of the original graph? On
an image this is quite easy because you can remove "every other vertex", but if you have an arbitrary
graph its pretty tricky to have a graph that's half the size capturing all the properties of the original
graph.

### Filtering
Multiplication in the Fourier domain corresonds to convolution in the time domain.

So a classical filter $h$ can be applied as $f_{out} = f_{in} (\epsilon) h(\epsilon)$

This also works in graph signal processing: $g_{out} (\lambda_l) = g_{in} (\lambda_l) h (\lambda_l)$.

We can filter directly on the vertex domain.

Let $N(i, j)$ be the set of vertices within a k-hop local neighbourhood. A filtering is the average
of all its neighbouring vertices, eg:

$g_{out}(i) = b_i, _i g_{in} (i) + \sum_{j \in N(i, k)} b_i, _j g_{in} (j)$

We could do a random walk in the graph. This is similar to what we do in deep learning - you
can think theory take the fourier transform of the signal and the apply the convolution operator
and take the inverse transform back, but we don't do that because on GPUs its far more efficient to
just apply convolution as a sliding window.

We could imagine this in terms of filtering an image. For instance, when we just take a
Gaussian filter, the degree of smoothing is completely invariant to frequency - it just
removes all the high frequency content, even if that content was actually describing
useful information.

But we can also look at an image as a graph. For instance, we connect not only to the top,
bottom, left and right neighbouring pixels, but also diagonally and then filter on the graph,
this produces a much more interesting denoising.

**Convolutions**: Classical definition is $(f * h)(t) = \int_R f(\tau)h(t - \tau) d \tau$

If we want to generalize this to graphs, we have a problem, due to the term $h(t - \tau)$. $h$
is our convolutional filter and this is transferred around. However, we don't know what translation
on a graph means. So this cannot be applied directly.

There is a way around it.

We cannot define the convolutional operator directly on a graph but we can do this
in the graph spectral domain:

$(g * h)(i) = \sum^{N - 1}_{i = 0} \hat g lambda_l \hat h(\lambda_l) u(i)^l$

**Translations**: Cannot be generalized

We can view trnaslation as doing a convolution with a delta centered at $v$

$(T_v g)(t) = \sqrt N (g * \delta_v)(t)$

The translation operator $Tv : R^N \to R^N$ becomes $(T_v g)(i) = \sqrt N (g * \delta_v)(i)$
for center vertex $n$ where $\delta_v(i) = \delta(v - i)$ (dirac delta function).

What does translation even mean? We could have a graph that represents the state
of minnesota and the node values are the amount of traffic flowing at each intersection.

If we have a signal defined at a certain point, we can translate it using a convolution
to find out what it would look like in other points.

**Modulation**: Modulation is frequency shift. You can do this classically by applying
an exponential to the originan function. $e^{2 \pi i w t} f(t)$.

This is a translation in the fourier domain.

For graphs, we can just replace the complex exponential with an eigenvector
of $L$:

$$ (M_l g)(i) = \sqrt N u(i)^(l) g(i) $$

This is not exactly a translation in the graph spectral domain but if $hat g$ is localized
aroudn 0, then $(M_l g)$ is localized around $\lambda_l$.

**Dialtion**: This is a scaling operator for a signal:

$(D_sf)(t) = \frac{1}{s} f(\frac{t}{s})$

This can't be generalized due to the posibility that $\frac{i}{s} \not \in V$, so
we need to use the fourier domain instead.

$D_s f(\epsilon) = \hat f(s \epsilon)$

We just multiply the frequency by the scalar $s$.

Assuming that we have a kernel $\hat g : R_+ \to R$ then the dilation for graphs is
$D_s g(\lambda_l) = \hat g (s \lambda_l)$ - dilation for graphs requires
$\hat g$ to be defined on the entire real line.

And we can do the same thing in the spectral domain for a graph, just multiply
by $s$.

**Downsampling**: For images we can downsample by just removing every other vertex.

But how do we do that with a graph without losing gometric information? If two vertices
are connected in a certain way, after doing pooling we still want to maintain that
connectivity and preserve the geometric stucture.

For many multiscale transforms we require successively coarser versions of the
original graph that preserve properties such as:
 - Geometric structure
 - Connectivity
 - Graph spectral distribution
 - Sparsity

Coarsening the graph boils down to finding a reduced set of vertices and assigning
edges and weights.

Eg: Coarsening with bipartite graphs: We can divide all the vertices into two sets
and determine the cut.

**Graph wavelets**: A wavelet is a feature detection mechanism:

$\psi (t) = \frac{2}{\sqrt{3 \sigma} \pi^{\frac{1}{4}} (1 - \frac{t^2}{\sigma^2}) e^{\frac{-t^2}{2 \sigma^2}}$

If we do a convolution with this, it will capture different features or edges in the image.

Wavelets can localize a signal in both time and frequency - whereas the fourier transform
can only handle frequency (but don't tell you where in the signal that frequency happened). Wavelets
can tell you where the frequency happened.

Wavelets are also possible on graphs. This is possible in the graph spectral domain
through the spectral graph wavelet transform. You have a set of $K$ wavelets centered
at each vertex and a scaling function (translated low-pass kernel).

You multiply the eigenvalues of the Laplacian matrix by $\hat h$ then you translate
it to somewhere using the delta function in the spectral domain.

You get a matrix that contains information about how much high frequency signal
you have at this area in the graph.

For instance, Minnesoda - red represents high traffic values and blue represents
low traffic values - once you go to the east there is a sudden discontinuity
in the graph.

The graph fourier transform allows you to know where the sudden discontinuity happens
and gives you information about what frequency type this discontinuity has. This applies
to any type of graph - if you have a graph that describes a sensor network you can understand
where something abnormal is happening.

## Application
Deep learning is possible but DL only works on Euclidean data. We assume that we can represent
the signal as a vector and each feature point is related to another feature by some euclidean distance.

Dimensionality reduction is another classic ML task that is done directly on graph
structured data by using graph signal processing. How can you generalize this?

Consider the Graph CNN. This is a normal CNN, except that instead of feeding it tensors
you can feed it any undirected graph. The graph then goes through learned spectral filters
which learn convolutions with respect to the graph struture. We can then pass it through
ReLU and coarsening (pooling) - this will hopefully represent the coarser version of
the graph which hopefully represents the structure.

Classical DL is a Graph CNN which has no edges at all. A Graph CNN is more general and
applies to the structure that your graph has.

**Graph PCA**: Learn the latent subspace that respects the graph structure. We have
some matrix $X$ and want to learn a lower-rank matrix $U$ which represents the underlying structure.

We can add two extra regularization terms:
 - *G-TV*: Graph total variation: Encodes some kind of regularization that the graph
           signal should be piecewise constant between samples. If you have an image
           and you represent each pixel as a vertex, then usually in the image you have a
           piecewise constant.
 - *G-TV*: Tikhonov graph regularizer: There should be smoothness between the features
           with respect to some graph structure. It could be that your feature is
           word2vec and maybe there is some assumption that some features should be
           closer to each other than others and Graph-PCA preserves this while lowering
           dimensionality.


**Computer Vision**: We have a lot of 3D geometric data, but most approaches are
                     tried "as is" on 3D geometric data. We basically just force the point
                     cloud to be represented as an image.

This has problems: The topological structure may be broken or details - when you convert as
                   an image you can only represent what the camera "sees". If you represented
                   this as a graph instead you could keep track of what was behind. The pixels
                   also change quite a lot when changing bose or deforming the object.

However if you represent as a graph then the graph has no concept of rotation, so we can
still model the same thing invariant to rotation.

## Conclusion
### Graph Signal Processing
The Graph Laplacian $L = D - W$ is central to graph signal processing.

We can interpret the eigenvalues of $L$ as a frequency.

There are several genearlized operators such as filtering and this is good at capturing
structural properties.

### Applications
 - Images and video
 - Represent object structure as a graph
 - Deep learning on graphs
 - Create robust methods by denoising signals on graphs

