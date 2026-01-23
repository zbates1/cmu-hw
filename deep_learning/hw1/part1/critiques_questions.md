for sigmoid backward: 6.1.2 Sigmoid Backward Equation
Backward propagation helps us understand how changes in pre-activation features Z affect the loss, given
how changes in post-activation values A affect the loss.
dL
dZ = sigmoid.backward(dLdA) (13)
= dLdA ⊙ ∂A
∂Z (14)
= dLdA ⊙(ς(Z) −ς2(Z)) (15)
= dLdA ⊙(A −A ⊙A) (16)

The above was the documentation, yet you wrote: dLdA * A * (1 - A)

-----------------------------------------------------------------

how did you come to tanh backward? 
-----------------------------------------------------------------

a little confused how you got the backward function for sigmoid

-----------------------------------------------------------------

dLdBW ∂L/∂γ matrix 1 ×C how changes in γ affect loss
dLdBb ∂L/∂β matrix 1 ×C how changes in β affect loss
dLdZ ∂L/∂Z matrix N ×C how changes in inputs affect loss
dLdNZ ∂L/∂ ˆZ matrix N ×C how changes in ˆZ affect loss
dLdBZ ∂L/∂  ̃Z matrix N ×C how changes in  ̃Z affect loss
dLdV ∂L/∂(σ2) matrix 1 ×C how changes in (σ2) affect loss
dLdM ∂L/∂μ matrix 1 ×C how changes in μ affect loss
Figure L: Batchnorm Topology
Note: In the following sections, we are providing you with element-wise equations instead of matrix equa-
tions. As a deep learning ninja, please don’t use for loops to implement them – that will be extremely
slow!
Your task is first to come up with a matrix equation for each element-wise equation we provide, then im-
plement them as code. If you ask TAs for help in this section, we will ask you to provide your matrix
equations.
10.1.1 Batch Normalization Forward Training Equations (When eval = False)
First, we calculate the mini-batch mean μ and variance σ2 of the current batch of data Z. μj and σ2j
represents the mean and variance of the jth feature. Zij refers to the element at the ith row and jth column
of Z and represents the value of the jth feature in ith sample in the batch.
33

-----------------------------------------------------
why is dldbz used as a parameter for the backward batchnorm
---------------------------------------------------------

### Answer:
BatchNorm forward computes: `BZ = γ * NZ + β` (scale and shift the normalized input)

`BZ` is the **output** of BatchNorm. During backprop, we receive `dL/dBZ` from the
layer above - this tells us how the loss changes w.r.t. BatchNorm's output.

We then use `dLdBZ` to compute all the other gradients working backward:
- `dLdBZ` → `dLdNZ` (gradient w.r.t. normalized input)
- `dLdNZ` → `dLdV` (gradient w.r.t. variance)
- `dLdNZ` → `dLdM` (gradient w.r.t. mean)
- All of these → `dLdZ` (gradient w.r.t. original input, which we return)

It's just the chain rule flowing backwards through the computation graph.

---------------------------------------------------------
Again in the norm section, you have this: dLdM = sum(dLdNZ * -(V+ε)^(-0.5), axis=0)

but it looks like the equation is supposed to be longer: ∂ ˆZi
∂μ = ∂
∂μ
[
(Zi −μ)(σ2 + ε)−1
2
]
i = 1,...,N (97)
= −(σ2 + ε)−1
2 − 1
2 (Zi −μ) ⊙(σ2 + ε)−3
2 ⊙
(
− 2
N
N∑
k=1
(Zk −μ)
)
(98)
∂L
∂μ =
N∑
i=1
∂L
∂ ˆZi
⊙ ∂ ˆZi
∂μ

### Answer:
The full equation (98) has two terms. The second term contains `Σ(Zk - μ)`.

But mathematically, **the sum of deviations from the mean is always zero**:
```
Σ(Zk - μ) = Σ Zk - N*μ = N*μ - N*μ = 0
```

So the entire second term vanishes, leaving only:
```
dLdM = sum(dLdNZ * -(V+ε)^(-0.5), axis=0)
```

This is a common simplification in BatchNorm backward derivations.