+++
title = 'A Coupled Cluster Neural Network Ansatz'
date = 2024-07-24T07:04:29-07:00
author = "A Nejati"
draft = false
math = true
+++

<!-- TODO: make notation consistent -->

<!-- TODO: works to review:
Unifying machine learning and quantum chemistry with a deep neural network for molecular wavefunctions.

Data‐Driven Acceleration of the Coupled‐Cluster Singles and Doubles Iterative Solver.

Symmetries and many-body excitations with neural-network quantum states.

-->

The main computational difficulty in variational methods is computing the energy integral over the entire domain, which is in practice intractable, therefore monte carlo integration techniques must be used. While VMC methods reduce the computational complexity of the method by avoiding evaluating a high-dimensional integral, they introduce the need for sampling a large number of system configurations, leading to slow convergence for larger molecules.

In contrast, in Coupled Cluster (CC) theory, we apply a similarity transform to the wavefunction and obtain a set of equations. The equations can be solved using e.g. quasi-Newton methods, and the solutions can be mapped back to the original basis. This similarity transform takes electron correlation effects into account through the use of an exponential cluster operator. This operator, when applied to a reference wavefunction (e.g. the Hartree-Fock determinant, in usual implementations), generates a hierarchy of excited determinants that effectively capture the correlated motion of electrons.
One of the most appealing aspects of coupled cluster theory is its size extensivity, meaning that the calculated energy scales correctly with the size of the system. This property is crucial for accurately describing chemical systems, especially when dealing with larger molecules or extended systems. Furthermore, the theory provides a clear pathway for improving accuracy through the inclusion of higher-order excitations, from the widely used CCSD (singles and doubles) to more advanced methods like CCSD(T), which are the "gold standard" of quantum chemistry.
Historically, CC has been used with Gaussian-type or Slater-type orbitals. In this work we set out to investigate if the use of CC with a Neural Network ansatz could offer improvements in accuracy without requiring excessively large bases.

For the specific neural network ansatz to use, we evaluate FermiNet [], a model that has been shown (in the VMC framework) to be able to compute accurate ground state wavefunctions. We also evaluate an attention-based ansatz.


## FermiNet

FermiNet [] is an explicitly antisymmetrized multi-layer neural network wavefunction defined on a fixed number of electrons, with an architecture which we will briefly describe. We first consider a simplified version of FermiNet that ignores both spin and electron pair correlations, which we call FermiNet-S1. Define a set of multi-electron functions $\varphi_i^k(\mathbf{x}\_j;\mathbf{x}\_1,\dots,\mathbf{x}\_{j-1},\mathbf{x}\_{j+1},\dots,\mathbf{x}\_n) = \varphi_i^k(\mathbf{x}\_j;\{\mathbf{x}\_{/j}\})$ where $\{\mathbf{x}\_{/j}\}$ is the unordered set of all electron states except $\mathbf{x}\_j$, and $i,j\in1,\dots,n$, the number of electrons. Define the following generalized Slater determinants:
$$
\psi^k(\mathbf{x}\_1,\dots,\mathbf{x}\_n) = \mathrm{det}\begin{bmatrix}
\varphi_1^k(\mathbf{x_1}) & \dots & \varphi_1^k(\mathbf{x}\_n) \\\\
\vdots & & \vdots \\\\
\varphi_n^k(\mathbf{x_1}) & \dots & \varphi_n^k(\mathbf{x}\_n)
\end{bmatrix} \tag{1}
$$

For ease of notation we have written $\varphi_i^k(\mathbf{x}\_j;\{\mathbf{x}\_{/j}\})$ as just $\varphi_i^k(\mathbf{x}\_j)$. These functions are themselves defined as multi-layer feedforward networks.

The final (complex-valued) amplitude is a linear combination of these determinants:

$$
\Psi = \sum_{k=1}^\chi \omega_k \psi^k 
$$

We may write the full wavefunction defined by FermiNet-S1 as:

<!-- TODO: clean this up -->
$$
|\Psi\rangle = \int_{\mathbb{R}^{3N}} \Psi(\mathbf{x}\_1,\dots,\mathbf{x}\_n)|\mathbf{x}\_1,\dots,\mathbf{x}\_n \rangle \mathrm{d}\mathbf{x}
$$

<!-- TODO: insert energy hamiltonian -->

We perform the variational minimization over the parameters of the networks and $\{\omega_k\}$. This defines a mapping from the electron position state space to amplitudes. 
### Coupled Cluster Formulation

The coupled cluster wavefunction is given as:

$$
|\Psi_\mathrm{CC}\rangle=\exp(\hat{T})|\Psi\rangle
$$

With the *cluster operator* $\hat{T}$ given as:

$$
\hat{T} = \hat{T}\_1 + \hat{T}\_2 + \dots + \hat{T}\_\mu
$$

Where each $\hat{T}\_m$ represents $m$-level excitations. here, $\mu$ is the order of the coupled cluster operator. A common choice is $\mu=2$, in which case we obtain the version of CC that takes single- and double-excitations into account, called Coupled Cluster Singles Doubles (CCSD). It can be shown that for two-electron systems, CCSD can provide an *exact* solution. For (N>2)-electron systems, CCSD is not exact, nevertheless in practice it usually provides a very good approximation to the electronic ground state. We can define the cluster operator explicitly in the second-quantization formulation, following [], as follows. For the single-excitation term,
$$
\hat{T}\_1 = \sum_\alpha^\mathrm{occ}\sum_\rho^\mathrm{vir}t_\alpha^\rho \hat{a}\_\rho^\dagger \hat{a}\_\alpha
$$

With $\hat{a}\_\rho^\dagger$ and $\hat{a}\_\alpha$ being the fermionic creation and annihilation operators, respectively. $\mathrm{occ}$ represents the set of ground-state-occupied (hole) states, and $\mathrm{vir}$ represents the set of unoccupied (electron) states. The coefficients $t_\alpha^\rho$ are parameters to be determined. 

For the double-excitation term,
$$
\hat{T}\_2 = \frac{1}{4}\sum_{\alpha,\beta}^\mathrm{occ}\sum_{\rho,\sigma}^\mathrm{vir}t_{\alpha\beta}^{\rho\sigma} \hat{a}\_\rho^\dagger \hat{a}\_\sigma^\dagger \hat{a}\_\beta \hat{a}\_\alpha
$$

Thus we see that the excitation terms represent operators that pull electrons out of ground states and place them in excited states, while keeping the total number of electrons constant.

For CCSD, the full operator is then:
$$
\exp(\hat{T}) = \hat{1} + \hat{T}\_1 + \hat{T}\_2 + \frac{1}{2}\hat{T}\_1^2 + \hat{T}\_1\hat{T}\_2 + \frac{1}{6}\hat{T}\_1^3 + \frac{1}{2}\hat{T}\_2^2 + \dots
$$


### Coupled Cluster FermiNet

We wish to apply this cluster operator to FermiNet. To do so, we must express it in the second quantized formulation and give meaning to the creation and annihilation operators. There are a few things to note. First, in fermionic second quantization, it is common to choose a fixed set of (antisymmetric) basis states, then rewrite the wavefunction in this basis. When the wavefunction is a slater determinant of orthogonal single-particle orbitals, it is common to choose the orbitals themselves as a basis. However, here, the orbitals are neither orthogonal nor fixed (as the parameters of the network can vary), complicating the use of this basis state. Second, FermiNet is defined for a fixed number of electrons, however creation (annihilation) operators increase (decrease) the number of electrons. We address both of these issues separately.

#### Particle Number
To represent FermiNet in second quantization where operators are typically allowed to change particle numbers, one option is to define an extension to the architecture where the number of input electrons is not fixed. Alternatively, and more simply, we can make use of the fact that our cluster operator only has terms of the form e.g. $\hat{a}\_\rho^\dagger \hat{a}\_\alpha$ that keep the total number of particles constant. Thus we only need to define the action of these particle number-preserving operators on the network.

Define the operators $\hat{a}\_\rho^\dagger \hat{a}\_\alpha$ as follows, based on their action on the individual slater determinants $\psi^k$:

$$
\begin{aligned}
\hat{a}\_\rho^\dagger \hat{a}\_\alpha |\psi^k\rangle &= \hat{a}\_\rho^\dagger \hat{a}\_\alpha \sum_{\mathbf{x}\_1,\dots,\mathbf{x}\_n} \psi^k(\mathbf{x}\_1,\dots,\mathbf{x}\_n)|\mathbf{x}\_1,\dots,\mathbf{x}\_n \rangle \\\\
&= \hat{a}\_\rho^\dagger \hat{a}\_\alpha \sum_{\mathbf{x}\_1,\dots,\mathbf{x}\_n} \mathrm{det}\begin{bmatrix}
\varphi_1^k(\mathbf{x_1}) & \dots & \varphi_1^k(\mathbf{x}\_n) \\\\
\vdots & & \vdots \\\\
\varphi_
\alpha^k(\mathbf{x_1}) & \dots & \varphi_\alpha^k(\mathbf{x}\_n) \\\\
\vdots & & \vdots \\\\
\varphi_n^k(\mathbf{x_1}) & \dots & \varphi_n^k(\mathbf{x}\_n)
\end{bmatrix}|\mathbf{x}\_1,\dots,\mathbf{x}\_n \rangle \\\\
  \\\\
&= \sum_{\mathbf{x}\_1,\dots,\mathbf{x}\_n} \mathrm{det}\begin{bmatrix}
\varphi_1^k(\mathbf{x_1}) & \dots & \varphi_1^k(\mathbf{x}\_n) \\\\
\vdots & & \vdots \\\\
\varphi_
\rho^k(\mathbf{x_1}) & \dots & \varphi_\rho^k(\mathbf{x}\_n) \\\\
\vdots & & \vdots \\\\
\varphi_n^k(\mathbf{x_1}) & \dots & \varphi_n^k(\mathbf{x}\_n)
\end{bmatrix}|\mathbf{x}\_1,\dots,\mathbf{x}\_n \rangle \\\\
\end{aligned}
$$

In other words, the action of such an operator is to remove the orbital function $\varphi_{i=\alpha}^k$ (represented by the corresponding row in the determinant) and replace it with a new orbital $\varphi_\rho^k$, while keeping the number of electrons (columns) constant.

Where does this new orbital come from? In the standard FermiNet, for each SD $k$, the number of orbitals is fixed and equal to the number of electrons. In this work, we extend this architecture to allow a number of *occupied* and *virtual* orbitals, but where only a subset of these orbitals at any time are used in the slater determinant calculation. The subset is always of size $n$. We call this architecture FermiNet-2Q(-S1). We write the new SDs as:
$$
\hat{\psi}^k(\mathbf{x}\_1,\dots,\mathbf{x}\_n;\pi_1,\dots,\pi_n) = \mathrm{det}\begin{bmatrix}
\varphi_{\pi_1}^k(\mathbf{x_1}) & \dots & \varphi_{\pi_1}^k(\mathbf{x}\_n) \\\\
\vdots & & \vdots \\\\
\varphi_{\pi_n}^k(\mathbf{x_1}) & \dots & \varphi_{\pi_n}^k(\mathbf{x}\_n)
\end{bmatrix}\tag{2}
$$

<!-- TODO: elaborate on FermiNet-2Q more. -->

With the new variables $\pi_{i\in1,\dots,n} \in 1,\dots,\nu$ representing a selection of orbitals, and $\nu = |\mathrm{occ}|+|\mathrm{vir}|$. Note that FermiNet is simply a special case of FermiNet-2Q where we set $\pi_i=i$. Another way of looking at it is that FermiNet describes the Fermi vacuum state of FermiNet-2Q.

We can trivially satisfy the canonical anticommutation relations by simply defining $\hat{a}\_\alpha\hat{a}\_\rho^\dagger = -\hat{a}\_\rho^\dagger \hat{a}\_\alpha$. That is, swapping the order of the operators performs the same row substitution, just with a negated sign on the determinant. We can also define $\hat{a}\_\alpha\hat{a}\_\alpha^\dagger = 1$. It is important to note that in our formulation we are *not* defining the individual operators $\hat{a}\_\alpha^\dagger$ or $\hat{a}\_\alpha$, thus this implicit anticommutation still results in well-defined operators.

We can write out the operators $\hat{a}\_\rho^\dagger \hat{a}\_\sigma^\dagger \hat{a}\_\beta \hat{a}\_\alpha$ in a similar way, as removing orbitals $\alpha$ and $\beta$ and replacing them with orbitals $\sigma$ and $\rho$ in the corresponding rows. For sake of brevity we omit the explicit representation here, however refer to the appendix for a complete derivation.

Returning to (2), it may seem unnatural, at first glance, to include discrete variables as parameters in a neural network, even as non-learned parameters (as they are here). However, this is in fact standard, in the form of the *dropout* procedure that is often used during neural network training []. For a rough analogy: In dropout, the activations of randomly selected neurons are simply dropped ('turned off') at random while doing gradient descent iterations. In the quantum analog described here, one can picture a subset of neurons being 'on' and another subset being 'off', and then considering all possible single-neuron 'excitations' (which turn a normally-on neuron off and a normally off-neuron on), and also all possible double-neuron excitations.

#### Non-orthogonal Orbitals

We now turn our attention to the second problem, which is that in the standard formulation of FermiNet, the orbitals $\varphi_i^k$ are not fixed and not (typically) orthogonal. To address this, we take inspiration from a technique recently proposed in (Entwistle 2022) for PauliNet (), where an overlap term (representing non-orthogonality of the orbitals) is added to the neural network loss:
$$
\mathcal{L}\_\mathrm{overlap}(\mathbf{\theta}) = \sum_{i>j} \left(\frac{1}{1-|S_{ij}|} - 1\right)
$$

Where:
$$
S_{ij} = \mathrm{sgn} \left(\mathbb{E}\_i \left[\frac{\Psi_{\theta,j}(\mathbb{r})}{\Psi_{\theta,i}(\mathbb{r})} \right] \right) \times \sqrt{\mathbb{E}\_i \left[\frac{\Psi_{\theta,j}(\mathbb{r})}{\Psi_{\theta,i}(\mathbb{r})} \right] \mathbb{E}\_j \left[\frac{\Psi_{\theta,i}(\mathbb{r})}{\Psi_{\theta,j}(\mathbb{r})} \right]}
$$

See (Entwistle 2022) for details. The total loss is then:
$$
\mathcal{L}(\mathbf{\theta}) = \mathcal{L}\_\mathrm{energy}(\mathbf{\theta}) + \alpha \mathcal{L}\_\mathrm{overlap}(\mathbf{\theta})
$$

The parameter $\alpha$ can be tuned during training. It is set to a small value at the start of training to ensure finding low-energy states, and then gradually increased to make those states orthogonal.

### CCSD Hamiltonian

With FermiNet-2Q and the corresponding operators defined, we turn to the Hamiltonian, which we write in the second quantized form. The second quantized Hamiltonian is usually written as:
$$
\hat{H} = \sum_{\rho, \alpha} s_\alpha^\rho \hat{a}\_\rho^\dagger \hat{a}\_\alpha + \frac{1}{2}\sum_{\alpha,\beta,\rho,\sigma}P_{\alpha\beta}^{\rho\sigma} \hat{a}\_\rho^\dagger \hat{a}\_\sigma^\dagger \hat{a}\_\beta \hat{a}\_\alpha
$$

Note the similarity of these terms to the single- and double-excitation operators, with the difference here being that $s$ and $P$ are determined:
$$
s_\alpha^\rho = \int \psi_\rho^*(\mathbf{x}) \left( -\frac{\hbar^2}{2m} \nabla^2 + V_{\text{ext}}(\mathbf{x}) \right) \psi_\alpha(\mathbf{x}) \, d\mathbf{x}
$$

$$
P_{\alpha\beta}^{\rho\sigma} = \int \int \psi_\rho^*(\mathbf{x}\_1) \psi_\sigma^\*(\mathbf{x}\_2) V(\mathbf{x}\_1, \mathbf{x}\_2) \psi_\beta(\mathbf{x}\_2) \psi_\alpha(\mathbf{x}\_1) \, d\mathbf{x}\_1 \, d\mathbf{x}\_2
$$

Where $V(\mathbf{x}\_1, \mathbf{x}\_2)$ is the interaction potential between electrons at positions $\mathbf{x}\_1$ and $\mathbf{x}\_2$.

<!-- Thus we can write:

$$
|\Psi\rangle = \sum_{\mathbf{x}\_1,\dots,\mathbf{x}\_n} \sum_{k=1}^\chi \omega_k \psi^k(\mathbf{x}\_1,\dots,\mathbf{x}\_n)|\mathbf{x}\_1,\dots,\mathbf{x}\_n \rangle
$$

A key point to note is that these basis functions themselves are varying, however for the time being we assume they are given as constant.
-->


Electronic excited states in deep variational
Monte Carlo. M. T. Entwistle 1,5, Z. Schätzle 1,5, P. A. Erdman 1
, J. Hermann1 &
F. Noé