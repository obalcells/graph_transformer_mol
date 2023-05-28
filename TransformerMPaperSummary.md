# Transformer M
## Representation of the graph
$$X \in \mathbb{R}^{n \times d}$$
where n number of atoms, d number of featues.

### 2 Dimensional Graphs:
$M^{2D} = (X, E)$, where $e_{(i,j)} \in E$ denotes edge feature, if present.
### 3 Dimensioal Graphs:
$M^{3D} = (X,R)$ where $R={r_1,...,r_n} and $r_i \in \mathbb{R}^3$.
We want model that can take either $M^{2D}$ or $M^{3D}$.
### Architecture:
Stacked transformer layers
transformer layer consists of two layers itself
x->
x + selfattention(norm(x))
x + feed-forward(norm(x))
both layers using normalization and skip connections.
## Encoding
The reason why the same layers stack can pass both 2D and 3D graphs is because in both cases the nodes are in the same space.
The fact, that the nodes are different will be accounted for in the attention bias. 

### Encoding pair-wise relations in E
Two terms used.
First encode lenght of shortest path between i and j
Learnable scalar $\Phi^{SPD}_{i,j}$. These biases are not shared across layers of heads. 
Now encode the shortest path itself. It is unique in most cases, where it is not just pick one.
Denote shortest path between i and j as $SP_{(i,j)} = (e_1, ..., e_N)$ then the edge encoding is defined to be 
$$\Phi^{Edge} _{(i,j)} = \frac{1}{N}\sum_{n=1}^{N} e_n(w_n)^T$$
Here $w_n$ are again learnable scalars that are not shared across heads nor layers. To make sure that this is well defined even during interference with a very large graph, one should cut them of at some point.
### Encoding pair-wise relations in R
First we choose a hyperparameter K.
Then for each pair $(i,j)$ we apply the Gaussian Basis Kerne function to the Euclidean distance between nodes i and j. 
$$\Psi^k_{(i,j)}= -\frac{1}{\sqrt{2\pi}|\sigma^k|}\exp{(-\frac{1}{2}(\frac{\gamma_{(i,j)}||r_i-r_j||+\beta_{(i,j)}-\mu_k}{|\sigma^k|}))}$$
$\sigma^k,\gamma^k, \beta^k$ and $\mu^k$ are learnable scalars. 
Now set $\Psi_{(i,j)} = [\Psi^1_{(i,j)}, ..., \Psi^n_{(i,j)}]$
Then the 3D distance encoding becomes
$$\Phi^{3D}_{ij} = \text{GELU}(\Psi_{(i,j)}W^1_D)W^2_D$$
Since $\Psi_{(i,j)}W^1_D$ is a vector matrix multiplication the result has dimension $1\times K$ The GELU activation does not affect the dimension.
After multiplikation with $W^2_D$ this yields a scalar. These matricies are spesific to the head and layer.
The GELU activation function can be understood as extention of the RELU activation function i a sense that is also accounts for a small range of negative inputs.
$$\text{GELU}(x) \approx 0.5x(1+\tanh[\sqrt(\frac{2}{\pi}(x+0.044715x^3)]$$
See image [here](https://medium.com/@shauryagoel/gelu-gaussian-error-linear-unit-4ec59fb2e47c).
Using this the self attention becomes:
$$A(X) = \text{softmax}(\frac{XW_Q(XW_K)^T}{\sqrt{d}} + \Phi^{SPD} + \Phi^{Edge} + \Phi^{3D})$$
## Centrality encoding
### Encodign atom-wise structural information in E
Define $$\Psi^{Degree} = [\Psi_1^{Degree}, ..., \Psi_n^{Degree}]$$
Where $\Psi_i^{Degree}$ is a d diomensional learnable vector.
### Encoding atom-wise structural information in R
Define the centrality encoding of the i-th atom as
$$\Psi_i^{\text{Sum of 3D distance}} = \sum_{i=1}^{n} \Psi_{(i,j)}W^3_D$$
Where $W^3_D$ is a learnable matrix of dimension $K \times d$.
Here \Psi_{(i,j)} is the shortest path encoding of the shortest path between i and j as obtained above.

### Encoding structural information in Transformer M
The centrality encoding is just added to the input $X^{(0)}$ of the network.
$$X^{(0)} = X + \Psi^{Degree} + \Psi^{\text{Sum of 3D distance}}$$
