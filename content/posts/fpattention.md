+++
title = 'Long Context Transformers with Finely Crafted State Spaces'
date = 2025-08-02T07:01:49-07:00
author = "A Nejati"
draft = false
math = true
+++

<!-- 
### Abstract
Linear-time transformers have emerged as a promising solution to the quadratic cost of standard self-attention, but existing methods face a difficult trade-off. Stochastic approaches introduce variance, while deterministic polynomial kernels like Power Attention offer only coarse, exponential control over their state size, limiting their practicality. We introduce Factorized Polynomial Attention (FPA), a new attention mechanism that is both exact and offers fine-grained, continuous control over its state space. FPA constructs a degree-$n$ polynomial kernel by factorizing it into a product of $n$ inner products in lower-dimensional projected spaces. This formulation allows the state size to be adjusted by tuning the projection dimensions, smoothly navigating the trade-off between memory footprint and expressive power without introducing stochasticity. FPA generalizes several existing linear attention mechanisms, including Power Attention, and its factorized structure is amenable to efficient, hardware-friendly recurrent implementations with $O(L)$ complexity. We formally define the FPA kernel, analyze its invariant properties, and outline a path to efficient GPU execution, presenting a versatile framework for building powerful and scalable long-context transformers. -->


The ability of transformers to capture long‐range dependencies hinges on the Θ(L²) cost of softmax self–attention, which becomes prohibitive once the context *L* reaches millions of tokens [1]. To solve this, people have been trying two broad strategies:

**Reducing the number of pairwise interactions.** Examples include strided or windowed schemes [2, 3] and low–rank projections such as Linformer [4]. These methods preserve the softmax kernel but an issue that keeps coming up is degraded recall on tasks that require global context.

**Replacing the kernel itself.** Linearized attention rewrites the softmax kernel as an inner product of feature maps $k(q,k)=\varphi(q)^{\top}\varphi(k)$, giving an $\Theta(L)$ recurrence [5]. Random–feature variants such as Performer’s $\mathrm{FAVOR}^+$ [6] are unbiased but introduce Monte‑Carlo variance that decays only as $\mathcal{O}(m^{-1/2})$ with the number $m$ of features. Deterministic kernels avoid variance but fix the memory footprint: Power attention [7] and its TPOW implementation require a $\binom{d+p-1}{p}$–sized state, which grows rapidly for $p>2$.

Recent work [7, 8, 9] has pointed out that the challenge for long-context models is a fundamental trade-off between recall capacity and inference throughput, governed by the model's recurrent state size. While standard attention performs well at recall, its KV-cache grows linearly, making it memory-intensive. Conversely, efficient alternatives with fixed-size states exhibit degraded performance on recall-intensive tasks.

A promising direction is to hybridize mechanisms: for instance, the BASED architecture [9] combines a degree-2 Taylor approximation of softmax for global context with local sliding-window attention for precision. The work we propose here introduces a significant generalization of this polynomial approach. We observe that the Taylor approximation is just one specific instance of a degree-2 polynomial kernel. In our approach, we provide a single, unified mechanism to navigate the state-size vs. recall Pareto frontier. By adjusting the projection dimensions $\{d_\ell\}$, our work offers a continuous memory knob to tune model capacity without resorting to stochastic methods or combining disparate architectural components. Furthermore, our framework is compatible with hybrid designs, suggesting that its global, higher-order modeling capabilities can be complemented by local attention mechanisms to capture both long-range dependencies and fine-grained local interactions, forming a powerful new class of efficient transformers.

**Factorized Polynomial Attention (FPA).** We propose¹ an exact kernel that expresses a degree‑$n$ polynomial as a product of $n$ lower‑dimensional dot products:
$$k_{\text{FPA}}(q,k)=\prod_{\ell=1}^{n}\bigl((W^{(\ell)}q)^{\top}(W^{(\ell)}k)\bigr)$$
where the branch widths $d_\ell$ are user‑specified. Setting $n=1$ recovers linear attention; $n=d$ and $W^{(\ell)}=I$ recover Power attention of order $d$. Between these extremes, $\sum_\ell d_\ell$ acts as a **continuous memory knob** with no stochastic error. In addition, if desired, additional parameters can be eliminated if $W^{(\ell)}$ are fixed blocks. The outer‑product update factorizes into $n$ standard GEMMs, enabling GPU/TPU fusion.

In this work, we introduce FPA, discuss various special cases, briefly discuss efficient implementation, and also discuss its relationship with other work. To our knowledge, no published method delivers an exact polynomial kernel with arbitrarily adjustable state that is amenable to the fast‑weight update. Works closest in spirit (product kernels [10], Kronecker attention [11], higher‑order transformers [12]) either maintain quadratic complexity in $L$ or fix the state width. We leave detailed evaluation of FPA on sequence modeling tasks to subsequent papers.

## Background

Our work builds upon the foundations of sequence modeling, primarily drawing from the Transformer architecture and its subsequent linear-time variants. We provide a brief overview of the concepts essential for understanding Factorized Polynomial Attention.

**Transformers and Self-Attention.** At the core of the Transformer architecture [1] is the self-attention mechanism. Given a sequence of input embeddings represented by matrices for queries ($Q$), keys ($K$), and values ($V$), all in $\mathbb{R}^{L \times d}$ where $L$ is the sequence length and $d$ is the model dimension, standard self-attention computes the output matrix $O \in \mathbb{R}^{L \times d}$ as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

**Recurrent Neural Networks.**
Recurrent Neural Networks (RNNs) offer a contrasting approach to sequence modeling. An RNN processes a sequence token-by-token, maintaining a fixed-size hidden state $h_t$ that evolves over time:
$$h_t = f(h_{t-1}, x_t)$$
where $x_t$ is the input at timestep $t$. This recurrent nature allows RNNs to operate with a computational cost that is linear in the sequence length, $O(L)$.

**Linear Attention.**
Linear attention replaces the exponential kernel (softmax) in self-attention with a linear one. As shown by [5], this allows the attention mechanism to be expressed in both a parallel form for training and a recurrent form for efficient inference. Linear attention uses a feature map $\phi(\cdot)$ such that the attention output for a query $q_t$ can be written as:
$$o_t = \frac{\sum_{i=1}^{t} (\phi(q_t)^T \phi(k_i)) v_i}{\sum_{i=1}^{t} \phi(q_t)^T \phi(k_i)} = \frac{\phi(q_t)^T \sum_{i=1}^{t} \phi(k_i) v_i^T}{\phi(q_t)^T \sum_{i=1}^{t} \phi(k_i)}$$
By exploiting the associativity of matrix multiplication, and introducing causal masking, the sums can be computed incrementally. Letting $S_t = \sum_{i=1}^{t} \phi(k_i) v_i^T$ and $Z_t = \sum_{i=1}^{t} \phi(k_i)$, the state can be updated recurrently:
$$S_t = S_{t-1} + \phi(k_t)v_t^T \quad \text{and} \quad Z_t = Z_{t-1} + \phi(k_t)$$
This formulation achieves $O(Ld^2)$ complexity for parallel training and $O(L)$ for autoregressive inference, but with a state size of $O(d^2)$. However, the sequence modeling performance of simple linear attention often lags behind standard softmax attention, suggesting that the expressive power offered by the large state size of dot-product kernel is crucial.

**Power Attention.** To enhance the expressiveness of linear attention, recent work has explored replacing the simple dot product with a polynomial kernel [7]. The output for power attention is defined as:

$$
\mathrm{attn}\_{\text{pow}}^{p}(Q, K, V)_i = \sum\_{j=1}^{i}(Q_i^T K_j)^p V_j
$$

Here, $Q_i$ and $K_j$ are vector representations from the query and key matrices, respectively.

This can be viewed as a specific instance of linear attention where the feature map $\phi(\cdot)$ is a tensorization of the input vector, mapping it to a much higher-dimensional space. Specifically, the tensorization is the **TPOW operation**, defined as follows.

For a vector $x \in \mathbb{R}^d$, its p-th tensor power, $x^{\otimes p}$, is a tensor of rank $p$. The TPOW operation flattens this tensor into a single vector in $\mathbb{R}^{d^p}$. The elements of this vector are all possible products of p elements from the original vector $x$:

$$
\mathrm{TPOW}_p(x) = \mathrm{flat}(x^{\otimes p}) =
\begin{bmatrix}
    \vdots \\\\
    \prod\_{k=1}^{p} x\_{i\_k} \\\\
    \vdots
\end{bmatrix}
\quad \text{for } (i_1, \dots, i_p) \in \{1, \dots, d\}^p
$$


A key mathematical property of the TPOW operation is that the inner product of the TPOW-transformed vectors $q$ and $k$ is equivalent to their original inner product raised to the p-th power:

$$
\mathrm{TPOW}_p(q)^T \mathrm{TPOW}_p(k) = (q^T k)^p
$$

This identity allows power attention to be framed as an instance of linear attention by setting the feature map $\phi = \mathrm{TPOW}_p$.

TPOW as described above contains many redundant entries (identical monomial terms) and unnecessary computation. In the same work ([7]), this drawback was identified, and a more optimized kernel (TSPOW) was suggested as a way of eliminating the redundancy by only retaining the (appropriately scaled) upper-triangular elements of the tensor of monomials. However, this becomes a distinct operation from TPOW and requires its own separate implementation.

Increasing dimensions causes the state space to grow exponentially. For instance, with $p=2$, the state size scales with $O(d^4)$, and for $p=3$, it scales with $O(d^6)$, significantly increasing the model's memory capacity. While powerful, this monolithic expansion can be computationally demanding and offers only coarse control over the state space size. Our work in this paper is directly motivated by the need for a more granular and efficient method to control the trade-off between model expressiveness and computational cost.

In the approach we introduce here, instead of taking the tensor product of the input vector with itself, we take the tensor product of multiple *projections* of the input vector into spaces of varying dimension. This offers a way to tune the state to any desired size while remaining exact and compatible with recursive implementations of linear/power attention-type with linear sequence length memory requirements.

## Factorized Polynomial Attention (FPA)

Here we formally describe FPA. FPA defines a kernel function by first projecting an input vector into several lower-dimensional spaces and then combining them with a product operation.

### The FPA Kernel

The kernel $\phi_{\text{FPA}}$ is constructed in two steps: a projection step and a product step.

First, an input vector $z \in \mathbb{R}^{d_{in}}$ is projected into $n$ separate feature vectors $z^{(1)}, z^{(2)}, \dots, z^{(n)}$ using distinct projection matrices. Each projection is defined as:
$$z^{(i)} = W^{(i)}z$$
where $W^{(i)} \in \mathbb{R}^{d_i \times d_{in}}$ is either a fixed or trainable weight matrix for the $i$-th projection, and the resulting vector $z^{(i)}$ has dimension $d_i$. Next, these projected vectors are combined using the Kronecker product ($\otimes$). The final feature map is the Kronecker product of all the projected vectors:

$$\phi_{\text{FPA}}(z) = z^{(1)} \otimes z^{(2)} \otimes \dots \otimes z^{(n)} = \bigotimes_{i=1}^{n} W^{(i)}z$$

The output is a vector in a high-dimensional space $\mathbb{R}^{D}$, where $D = \prod_{i=1}^{n} d_i$. In the case where $W^{(i)} = I$, this reduces to Power Attention with power $n$ and state space size $d^n$. As before, in the case where $n = 1$ and $W^{(1)}=I$, this reduces to standard linear attention. Importantly, by appropriate choice of $d_i$, we can finely tune the size of the state space.

An important note is that the projection matrices $W^{(i)}$ are unique to each attention head. Indeed, the matrices need not even have the same shape across different heads. We can make use of this property to show that the optimized TSPOW kernel is also a special case of FPA, see Appendix.

### Relationship to Linear Attention
A key basic property of the Kronecker product allows computing this kernel without explicitly constructing the state space, as with power attention. The inner product of two transformed vectors in the high-dimensional space simplifies to a product of inner products in the lower-dimensional projected spaces.

For two vectors $q$ and $k$:

$$
\begin{align*}
\phi_{\text{FPA}}(q)^T \phi_{\text{FPA}}(k) &= \left(\bigotimes_{i=1}^{n} (W^{(i)}q)\right)^T \left(\bigotimes_{i=1}^{n} (W^{(i)}k)\right) \\\\
&= \prod_{i=1}^{n} \left( (W^{(i)}q)^T (W^{(i)}k) \right)
\end{align*}
$$

Substituting this into the linear attention framework gives the final formula for FPA:

$$
\mathrm{attn}\_{\text{FPA}}(Q, K, V)_i = \sum\_{j=1}^{i} \left( \prod\_{l=1}^{n} ((W^{(l)}Q_i)^T (W^{(l)}K_j)) \right) V_j
$$

### Tunability of the State Space
FPA exposes two orthogonal knobs: **Branch count** $n$ (kernel degree) and **branch widths** $\{d_k\}$ (state size), tunable to any value between $d$ and $d^n$.

FPA has several advantages. It is exact but without blow-up, offering a deterministic kernel with a finely adjustable state size. Special cases of FPA, such as block‑identity or fixed orthogonal $W^{(k)}$ variants, add few or zero trainable parameters. It is hardware-friendly, with each branch being a standard GEMM; the outer product factorizes into $n$ small updates, compatible with tensor‑core pipelines, tensor‑train and sketch compression. The inductive bias provided by multiplicative feature interactions is useful for arithmetic, symbolic and some vision tasks.


## Invariant Properties of FPA

FPA inherits several algebraic symmetries from its construction. Understanding these invariances clarifies both its expressive biases and the kinds of transformations that can be absorbed without changing the attention scores.

1.  **Exchange symmetry.** Because each factor in the FPA kernel is a symmetric bilinear form, the kernel satisfies $k_{\text{FPA}}(q,k)=k_{\text{FPA}}(k,q)$. Consequently, the attention operates on unordered pairs of tokens.

2.  **Branch permutation symmetry.** Multiplication is commutative, so permuting the list $\{W^{(1)},\dots,W^{(n)}\}$ leaves the kernel unchanged. This implies that networks need not track branch order, which simplifies implementation when $n$ is large.

3.  **Common isometry group.** Define the quadratic forms
$$
G^{(\ell)}=(W^{(\ell)})^{\top}W^{(\ell)}
$$
Let $\mathcal{G} = \{U \in O(d_{\text{in}}) \mid U^{\top}G^{(\ell)}U=G^{(\ell)}; \forall \ell\}$. Then $k_{\text{FPA}}(q,k)$ is invariant under simultaneous rotations $(q,k)\mapsto(Uq,Uk)$ for any $U\in\mathcal{G}$. When all $G^{(\ell)}$ are proportional to the identity, $\mathcal{G}$ equals the full orthogonal group and the kernel is isotropic; otherwise $\mathcal{G}$ shrinks to the intersection of the individual stabilizers.

4.  **Scaling degeneracy.** Re‑scaling a single projection $W^{(\ell)} \mapsto \alpha W^{(\ell)}$ multiplies the kernel by $\alpha^2$. A global renormalization therefore renders absolute scales irrelevant; only the relative norms across branches matter.

5.  **Factor‑wise linearity.** Holding all but one factor fixed, the kernel is linear in that factor. This separability is the key to its efficient fast‑weight implementation.

## Special Cases and Sub‑Families of FPA

The general formulation admits numerous interesting sub-families that trade expressiveness against parameter count and invariance. Table 1 summarises some useful ones. As mentioned, Power attention and its TSPOW variant emerge as special cases of FPA, but other novel subtypes emerge as well, providing a large design space to explore.

**Table 1: Representative sub-families of FPA.** All remain positive-definite kernels; each imposes different structural sparsity on the coefficient tensor.

| **Name** | **Constraint on $G^{(\ell)}$** | **Implications** |
| :--- | :--- | :--- |
| Linear attention | $n = 1$ | Original fast-weight kernel with $O(d)$ state. |
| Power attention | $n=p, W^{(\ell)} = I$ | Homogeneous degree-$p$ kernel with $d^p$-sized state (TPOW). |
| Scaled-shared | $G^{(\ell)}=\alpha_\ell G_0$ | Reduces to power attention up to a scalar; illustrates parameter redundancy. |
| Commuting set | $[G^{(\ell)},G^{(m)}]=0$ | Separates direction vs. branch weighting; Simultaneously diagonalizable. |
| Diagonal factors | $G^{(\ell)}=\operatorname{diag}(g^{(\ell)})$ | Monomials that never mix coordinates; feature map size $\prod_\ell d_\ell$. |
| Rank-1 factors | $G^{(\ell)}=a^{(\ell)}a^{(\ell)\top}$ | Minimal-parameter degree-$n$ kernel acting along $n$ learned 1-D directions. |
| Orthogonal blocks | $\operatorname{tr}(G^{(\ell)}G^{(m)})=0$ for $\ell \neq m$ | Independent sub-space interactions; useful for block-separable structure. |
| Projector factors | $G^{(\ell)2}=G^{(\ell)}$ | Measures overlap inside $n$ chosen sub-spaces (idempotent). |
| Coordinate Select | $G^{(\ell)}=\operatorname{diag}(g^{(\ell)} \in \{0, 1\}^d)$| Parameter-free projection that forces interactions only between selected subsets of features. A sub-case of Diagonal and Projector factors. |
| TSPOW | fixed block-identity $W^{(\ell)}$ | Eliminates monomial redundancy of TPOW while retaining exactness; obtained by setting mutually disjoint coordinate blocks. |

Notably, some of the special cases lend themselves to highly efficient implementations. For instance, the Coordinate Select case can simply be rewritten as choosing a subset of elements of the input vector.


## Efficient Implementation
A naive implementation of FPA that explicitly materializes the high-dimensional feature maps $\phi_{\text{FPA}}(q)$ and $\phi_{\text{FPA}}(k)$ would not be feasible; the resulting vectors, of dimension $D = \prod_{i=1}^{n} d_i$, would incur prohibitive memory and I/O costs, creating a bottleneck that limits arithmetic intensity. This issue is analogous to the challenge posed by the intermediate $L \times L$ attention matrix in standard softmax attention. To overcome this, we adopt an I/O-aware approach inspired by FlashAttention [13] and its adaptation for Power Attention [7]. By fusing operations into a single GPU kernel, all intermediate computations can be performed on-chip in fast SRAM, avoiding costly reads and writes to HBM. The factorized structure of the FPA kernel is particularly amenable to this strategy. FPA can be implemented using two complementary strategies depending on sequence length.

**Parallel (Tiled) Form.** For short to moderate sequence lengths, FPA is implemented as a single fused kernel that computes the attention output block-wise. The inputs $Q, K, V$ are tiled, and for each block of queries and keys, the kernel computes the FPA score matrix on-chip without forming the full $\phi_{\text{FPA}}$ vectors. Instead, it directly computes the product of inner products:
$$S_{\text{block}} = \prod_{\ell=1}^{n} \left( (W^{(\ell)}Q_{\text{block}})^T (W^{(\ell)}K_{\text{block}}) \right)$$
This involves $n$ independent, parallelizable matrix multiplications whose results are combined with an element-wise product reduction. The resulting score block is then used to compute the output, which is written back to HBM.

**Recurrent (Chunked) Form.** For very long sequences, FPA's equivalence to linear attention permits a recurrent formulation with $O(L)$ complexity. We utilize a chunked parallel algorithm. The state update rule, $S_t = S_{t-1} + \phi_{\text{FPA}}(k_t)v_t^T$, is computed in segments. The outer product $\phi_{\text{FPA}}(k_t)v_t^T$ is computed efficiently by exploiting the Kronecker product structure of $\phi_{\text{FPA}}(k_t)$, again avoiding materialization of the high-dimensional state. This dual-formulation ensures that FPA can be implemented efficiently across all sequence length regimes, combining the parallelism of attention with the linear-time complexity of RNNs.

---
### Footnotes
¹ Code at `github.com/anj1/fpa`.

---
## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

[2] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*, 2020.

[3] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. *arXiv preprint arXiv:1904.10509*, 2019.

[4] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. *arXiv preprint arXiv:2006.04768*, 2020.

[5] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In *International conference on machine learning*, pages 5156–5165. PMLR, 2020.

[6] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. *arXiv preprint arXiv:2009.14794*, 2020.

[7] Carles Gelada, Jacob Buckman, Sean Zhang, and Txus Bach. Scaling context requires rethinking attention. *arXiv preprint arXiv:2507.04239*, 2025.

[8] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In *International conference on machine learning*, pages 9355–9366. PMLR, 2021.

[9] Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou, Atri Rudra, and Christopher Ré. Simple linear attention language models balance the recall-throughput tradeoff. *arXiv preprint arXiv:2402.18668*, 2024.

[10] Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, and Ruslan Salakhutdinov. Transformer dissection: a unified understanding of transformer’s attention via the lens of kernel. *arXiv preprint arXiv:1908.11775*, 2019.

[11] Hongyang Gao, Zhengyang Wang, and Shuiwang Ji. Kronecker attention networks. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pages 229–237, 2020.

[12] Soroush a d Rabusseau Omranpour, Guillaume and Reihaneh Rabbany. Higher order transformers: Efficient attention mechanism for tensor structured data. *arXiv preprint arXiv:2412.02919*, 2024.

[13] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. *Advances in neural information processing systems*, 35:16344–16359, 2022.

## Appendix

### TSPOW is a special case of FPA

Below we show that the *truncated symmetric power* (TSPOW) feature map of [7] of degree 2 can be reproduced exactly by an FPA kernel with two branches ($n=2$) and multiple heads, each head using a pair of carefully chosen (fixed, non-trained) linear projections $W^{(1)},W^{(2)}$.

#### Degree‑2 TSPOW, block size $b=2$
Let the input be
$$x=[x_1,\dots,x_d]^{\top},\qquad d=mb\;(m=\lceil d/2\rceil).$$
TSPOW keeps only the upper‑triangular monomials $x_i x_j$ with $i\le j$ and rescales mixed terms by $\sqrt{2}$:

| Kept Term | TSPOW Coefficient |
| :--- | :--- |
| $\lceil i/2\rceil = \lceil j/2\rceil$ | 1 |
| $\lceil i/2\rceil < \lceil j/2\rceil$ | $\sqrt{2}$ |
| $\lceil i/2\rceil > \lceil j/2\rceil$ | discarded |

**Head construction:**
For head $h \in \{0, \dots, m-1\}$ define its *anchor block*
$$S_h=\{2h+1, 2h+2\}.$$
Then set two branch projections:
* **Branch $W^{(1)}\_{h}$**: $2 \times d$ matrix that selects entries in $S_h$ (identity on those, zeros elsewhere).
* **Branch $W^{(2)}\_{h}$**: $d \times d$ matrix that keeps entries with index $\ge 2h+1$; multiplies the ones outside $S_h$ by $\sqrt{2}$; zeros everything with index $<2h+1$.

Formally, for column index $j$,
$$
W^{(2)}\_{h}[j,j]=
\begin{cases}
1, & j\in S_h\\\\[2pt]
\sqrt{2}, & j>2h+2\\\\[2pt]
0, & j<2h+1.
\end{cases}
$$
All off‑diagonal weights are 0, so each $W$ is just a diagonal mask.

**What each head contributes:**
For any two vectors $q,k$
$$
k_{\text{FPA},h}(q,k)=
\bigl((W^{(1)}\_{h}q)^{\top}(W^{(1)}\_{h}k)\bigr)
\bigl((W^{(2)}\_{h}q)^{\top}(W^{(2)}\_{h}k)\bigr)
$$

Expanding the dot‑products shows that each head emits exactly the monomials $x_i x_i$ and $x_i x_j\;(i<j)$ with $i\in S_h, j\ge i$. Because terms with $j<i$ are wiped out (their second factor is 0); and cross‑block terms get the desired $\sqrt{2}$ factor from $W^{(2)}\_{h}$. Summing all heads $h=0,\dots,m-1$ therefore enumerates every upper‑triangular pair once with the correct coefficient. That is, the $\text{TSPOW}\_{2 \times 2}$ feature map.

#### General degree‑2 TSPOW, arbitrary block size $b$
Let blocks be contiguous chunks of length $b$:
$$S_h=\{hb+1,\dots,(h+1)b\},\qquad h=0,\dots,m-1,\;m=\lceil d/b\rceil.$$
Define per‑head projections:

$$
W^{(1)}\_{h}: x \mapsto x|_{S_h}, \qquad
W^{(2)}\_{h}: x_j \mapsto
\begin{cases}
x_j,& j\in S_h,\\\\
\sqrt{2}\,x_j,& j>(h+1)b,\\\\
0,& j<(hb+1).
\end{cases}
$$
Again, head $h$ covers all pairs with (left) index in $S_h$ and right‑index $\ge$ that left index. The union over heads realizes the upper‑triangular mask of TSPOW.

#### Sketch for higher‑degree TSPOW ($p>2$)
$\textrm{TSPOW}\_p$ keeps monomials $x_{i_1}\dots x_{i_p}$ with a sorted index tuple $i_1\le\dots\le i_p$ and scales by $\sqrt{\text{mult}}$. We can extend the previous construction as follows. Use $n=p$ branches in FPA. Enumerate all non‑decreasing block‑index tuples $(h_1\le\dots\le h_p)$. For each tuple create one head. For branch $r$ in that head, $W^{(r)}$ selects the $b$ coordinates of block $S_{h_r}$. Coordinates whose block index is greater than $h_r$ are multiplied by $\sqrt{p!/\text{mult}}$ in exactly enough branches to reproduce the TSPOW scaling.

This construction produces every sorted index tuple once (no duplicates) and applies the right combinatorial scaling, while preserving the factorised (linear‑time) structure of FPA.

#### Remarks
Viewing TSPOW via FPA clarifies that many "special" polynomial kernels are just different block masks and scalings; they can therefore inherit the same hardware paths and optimisation tricks as generic FPA.

As stated before, however, this construction is only possible by choosing the projection matrices $W^{(i)}$ to have different shapes across different heads. In practice, this can complicate the implementation because different gpu threads will have to compute different branches, just as the implementation in [7] describes. There is no free lunch here; writing TSPOW in terms of FPA does not provide a free performant gpu implementation. Specialized code is still needed.