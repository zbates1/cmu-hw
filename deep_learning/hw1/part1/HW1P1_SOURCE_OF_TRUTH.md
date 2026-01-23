# HW1P1 Source of Truth - MyTorch Implementation

**Author: Zane**

## CRITICAL IMPLEMENTATION RULES

1. **DO NOT DELETE ANY ORIGINAL COMMENTS** - Keep all the original docstrings, hints, and comments from the starter code
2. **COMMENT OUT PLACEHOLDER CODE** - When there's placeholder code like `self.A = None  # TODO`, comment it out and add the implementation below it:
   ```python
   # self.A = None  # TODO
   self.A = actual_implementation
   ```
3. **PRESERVE ALL ARGUMENTS** - Never delete or modify function arguments
4. **PRESERVE STRUCTURE** - Maintain the exact structure of the original files
5. **NO AI/CLAUDE ATTRIBUTION** - Do not add any comments mentioning AI or Claude as author
6. **AUTHOR ATTRIBUTION** - If any authorship is written, it should say "Zane"
7. **KEEP BOILERPLATE AS MAP** - Leave original boilerplate code commented so it serves as a reference

## 1. Linear Layer [mytorch.nn.Linear]

### Shapes
- W: (Cout, Cin) - weight matrix
- b: (Cout, 1) - bias vector
- A: (N, Cin) - input batch
- Z: (N, Cout) - output batch

### Forward
```
Z = A @ W.T + ones @ b.T
```
Where `ones` is (N, 1) column vector of 1s.

### Backward
Given dLdZ (N, Cout):
```
dLdA = dLdZ @ W          # (N, Cin)
dLdW = dLdZ.T @ A        # (Cout, Cin)
dLdb = dLdZ.T @ ones     # (Cout, 1)
```

---

## 2. Activation Functions

### 2.1 Sigmoid
Forward:
```
A = 1 / (1 + exp(-Z))
```

Backward:
```
dLdZ = dLdA * (A - A * A)
     = dLdA * A * (1 - A)
```

### 2.2 Tanh
Forward (DO NOT use np.tanh):
```
A = (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))
```

Backward:
```
dLdZ = dLdA * (1 - A * A)
```

### 2.3 ReLU
Forward:
```
A = max(0, Z)   # use np.maximum(0, Z)
```

Backward:
```
dLdZ = dLdA * (A > 0)   # or np.where(A > 0, dLdA, 0)
```

### 2.4 GELU
Forward:
```
A = Z * 0.5 * (1 + erf(Z / sqrt(2)))
```

Backward:
```
dAdZ = 0.5 * (1 + erf(Z / sqrt(2))) + (Z / sqrt(2*pi)) * exp(-Z^2 / 2)
dLdZ = dLdA * dAdZ
```

### 2.5 Swish
Forward (with learnable beta):
```
sigma = 1 / (1 + exp(-beta * Z))
A = Z * sigma
```

Backward:
```
dAdZ = sigma + beta * Z * sigma * (1 - sigma)
dLdZ = dLdA * dAdZ

dAdbeta = Z * Z * sigma * (1 - sigma)
dLdbeta = sum(dLdA * dAdbeta)
```

### 2.6 Softmax
Forward (with numerical stability):
```
Z_stable = Z - max(Z, axis=1, keepdims=True)
exp_Z = exp(Z_stable)
A = exp_Z / sum(exp_Z, axis=1, keepdims=True)
```

Backward (per sample i):
```
Jacobian J[m,n]:
  if m == n: J[m,n] = A[m] * (1 - A[m])
  if m != n: J[m,n] = -A[m] * A[n]

dLdZ[i,:] = dLdA[i,:] @ J
```

---

## 3. MLP Models

### MLP0 (0 hidden layers)
Layers: [Linear(2,3), ReLU()]

Forward:
```
Z0 = layers[0].forward(A0)
A1 = layers[1].forward(Z0)
return A1
```

Backward:
```
dLdZ0 = layers[1].backward(dLdA1)
dLdA0 = layers[0].backward(dLdZ0)
return dLdA0
```

### MLP1 (1 hidden layer)
Layers: [Linear(2,3), ReLU(), Linear(3,2), ReLU()]

Forward:
```
Z0 = layers[0].forward(A0)
A1 = layers[1].forward(Z0)
Z1 = layers[2].forward(A1)
A2 = layers[3].forward(Z1)
return A2
```

Backward:
```
dLdZ1 = layers[3].backward(dLdA2)
dLdA1 = layers[2].backward(dLdZ1)
dLdZ0 = layers[1].backward(dLdA1)
dLdA0 = layers[0].backward(dLdZ0)
return dLdA0
```

### MLP4 (4 hidden layers)
Layers: [Linear(2,4), ReLU(), Linear(4,8), ReLU(), Linear(8,8), ReLU(), Linear(8,4), ReLU(), Linear(4,2), ReLU()]

Forward:
```
for i in range(len(layers)):
    A = layers[i].forward(A)
return A
```

Backward:
```
for i in reversed(range(len(layers))):
    dLdA = layers[i].backward(dLdA)
return dLdA
```

---

## 4. Loss Functions

### MSELoss
Forward:
```
SE = (A - Y) * (A - Y)
SSE = ones_N.T @ SE @ ones_C   # sum all elements
MSE = SSE / (N * C)
return MSE
```

Backward:
```
dLdA = 2 * (A - Y) / (N * C)
return dLdA
```

### CrossEntropyLoss
Forward:
```
# Softmax
exp_A = exp(A - max(A, axis=1, keepdims=True))  # numerical stability
softmax = exp_A / sum(exp_A, axis=1, keepdims=True)

# Cross-entropy
crossentropy = (-Y * log(softmax)) @ ones_C  # (N, 1)
sum_crossentropy = ones_N.T @ crossentropy   # scalar
mean_crossentropy = sum_crossentropy / N
return mean_crossentropy
```

Backward:
```
dLdA = (softmax - Y) / N
return dLdA
```

---

## 5. SGD Optimizer

### Without Momentum (mu = 0)
```
W = W - lr * dLdW
b = b - lr * dLdb
```

### With Momentum (mu != 0)
```
v_W = mu * v_W + dLdW
v_b = mu * v_b + dLdb
W = W - lr * v_W
b = b - lr * v_b
```

---

## 6. BatchNorm1d

### Forward (Training, eval=False)
```
# Mini-batch mean and variance
M = mean(Z, axis=0, keepdims=True)      # (1, C)
V = var(Z, axis=0, keepdims=True)       # (1, C) - use population var (ddof=0)

# Normalize
NZ = (Z - M) / sqrt(V + eps)

# Scale and shift
BZ = BW * NZ + Bb

# Update running stats
running_M = alpha * running_M + (1 - alpha) * M
running_V = alpha * running_V + (1 - alpha) * V
```

### Forward (Inference, eval=True)
```
NZ = (Z - running_M) / sqrt(running_V + eps)
BZ = BW * NZ + Bb
```

### Backward
```
dLdBb = sum(dLdBZ, axis=0, keepdims=True)      # (1, C)
dLdBW = sum(dLdBZ * NZ, axis=0, keepdims=True) # (1, C)

dLdNZ = dLdBZ * BW                              # (N, C)

dLdV = sum(dLdNZ * (Z - M) * -0.5 * (V + eps)^(-3/2), axis=0, keepdims=True)

dNZdM = -(V + eps)^(-1/2) - 0.5 * (Z - M) * (V + eps)^(-3/2) * (-2/N) * sum(Z - M, axis=0)
      = -(V + eps)^(-1/2)  # simplified since sum(Z-M)=0

dLdM = sum(dLdNZ * (-(V + eps)^(-1/2)), axis=0, keepdims=True) + dLdV * (-2/N) * sum(Z - M, axis=0)

dLdZ = dLdNZ * (V + eps)^(-1/2) + dLdV * (2/N) * (Z - M) + dLdM * (1/N)
```

Simplified backward from writeup equation (100):
```
dLdZ = dLdNZ * (V + eps)^(-1/2) + dLdV * (2/N) * (Z - M) + (1/N) * dLdM
```
