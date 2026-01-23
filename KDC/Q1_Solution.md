# Question 1: Unit Quaternions

> **Notation:** $Q = (q_0, \vec{q})$ where $q_0 \in \mathbb{R}$ is scalar, $\vec{q} \in \mathbb{R}^3$ is vector.
> **Multiplication:** $(a_0, \vec{a})(b_0, \vec{b}) = (a_0b_0 - \vec{a}\cdot\vec{b}, \ a_0\vec{b} + b_0\vec{a} + \vec{a}\times\vec{b})$

---

## Part 1 (2pts): Show QXQ* is Pure and Derive Rotation Formula

**Given:**
- $Q = (q_0, \vec{q})$ unit quaternion, so $q_0^2 + |\vec{q}|^2 = 1$
- $X = (0, \vec{x})$ pure quaternion
- $Q^* = (q_0, -\vec{q})$ conjugate

### Step 1: Compute QX

$$QX = (q_0, \vec{q})(0, \vec{x}) = \bigl(-\vec{q}\cdot\vec{x}, \ q_0\vec{x} + \vec{q}\times\vec{x}\bigr)$$

### Step 2: Compute (QX)Q*

Let $s = -\vec{q}\cdot\vec{x}$ and $\vec{v} = q_0\vec{x} + \vec{q}\times\vec{x}$

**Scalar part:**
$$sq_0 + \vec{v}\cdot\vec{q} = -q_0(\vec{q}\cdot\vec{x}) + q_0(\vec{x}\cdot\vec{q}) + \underbrace{(\vec{q}\times\vec{x})\cdot\vec{q}}_{=0} = 0$$

> *The cross product is perpendicular to* $\vec{q}$*, so the dot product vanishes.*

$$\therefore \quad \textbf{QXQ* is a pure quaternion} \quad \checkmark$$

**Vector part:**
$$\begin{aligned}
&(\vec{q}\cdot\vec{x})\vec{q} + q_0^2\vec{x} + q_0(\vec{q}\times\vec{x}) - q_0(\vec{x}\times\vec{q}) - (\vec{q}\times\vec{x})\times\vec{q} \\[6pt]
&= (\vec{q}\cdot\vec{x})\vec{q} + q_0^2\vec{x} + 2q_0(\vec{q}\times\vec{x}) - (\vec{q}\times\vec{x})\times\vec{q}
\quad \text{(since } \vec{x}\times\vec{q} = -\vec{q}\times\vec{x}\text{)}
\end{aligned}$$

Using BAC-CAB identity: $(\vec{q}\times\vec{x})\times\vec{q} = |\vec{q}|^2\vec{x} - (\vec{q}\cdot\vec{x})\vec{q}$

$$\boxed{(q_0^2 - \vec{q}\cdot\vec{q})\vec{x} + 2q_0(\vec{q}\times\vec{x}) + 2(\vec{x}\cdot\vec{q})\vec{q}}$$

### Step 3: Verify This is Rodrigues' Formula

For $Q = (\cos\frac{\theta}{2}, \sin\frac{\theta}{2}\hat{n})$ representing rotation by $\theta$ about axis $\hat{n}$:

| Expression | Simplifies to |
|------------|---------------|
| $q_0^2 - \|\vec{q}\|^2$ | $\cos\theta$ |
| $2q_0\|\vec{q}\|$ | $\sin\theta$ |
| $2\|\vec{q}\|^2$ | $1 - \cos\theta$ |

Substituting gives **Rodrigues' rotation formula**:
$$\vec{x}' = \cos\theta\,\vec{x} + \sin\theta(\hat{n}\times\vec{x}) + (1-\cos\theta)(\hat{n}\cdot\vec{x})\hat{n} \quad \checkmark$$

---

## Part 2 (2pts): Two-to-One Covering of SO(3)

**Claim:** For each $R \in SO(3)$, exactly two unit quaternions $\{Q, -Q\}$ represent $R$.

### Proof: Q and -Q Give the Same Rotation

$$(-Q)X(-Q)^* = (-1)(-1) \cdot QXQ^* = QXQ^* \quad \checkmark$$

### Proof: These Are the Only Two

Suppose $QXQ^* = PXP^*$ for all pure $X$. Then $(P^*Q)X = X(P^*Q)$.

Let $P^*Q = (r_0, \vec{r})$. The commutator is:
$$(r_0, \vec{r})(0, \vec{x}) - (0, \vec{x})(r_0, \vec{r}) = (0, \ 2\vec{r}\times\vec{x})$$

For this to vanish for all $\vec{x}$: $\vec{r} = \vec{0}$

Thus $P^*Q = (r_0, \vec{0})$ with $|P^*Q| = 1 \Rightarrow r_0 = \pm 1 \Rightarrow Q = \pm P$ ∎

---

## Part 3 (2pts): Operation Counts

### (i) Compose Two 3×3 Rotation Matrices

9 entries × (3 mult + 2 add) each:

| **Multiplications** | **Additions** |
|:-------------------:|:-------------:|
| 27 | 18 |

### (ii) Compose Two Quaternions

| Component | Mult | Add |
|-----------|:----:|:---:|
| Scalar $q_0p_0 - \vec{q}\cdot\vec{p}$ | 4 | 3 |
| Vector scaling $q_0\vec{p} + p_0\vec{q}$ | 6 | 3 |
| Cross product $\vec{q}\times\vec{p}$ | 6 | 3 |
| Final vector sum | — | 3 |
| **Total** | **16** | **12** |

### (iii) Apply Rotation Matrix to Vector

3 components × (3 mult + 2 add):

| **Multiplications** | **Additions** |
|:-------------------:|:-------------:|
| 9 | 6 |

### (iv) Apply Quaternion to Vector (QXQ*)

Two quaternion-pure products:

| **Multiplications** | **Additions** |
|:-------------------:|:-------------:|
| ~24 | ~17 |

### Summary Comparison

| Operation | Mult | Add | **Total** |
|-----------|:----:|:---:|:---------:|
| Matrix × Matrix | 27 | 18 | **45** |
| Quat × Quat | 16 | 12 | **28** |
| Matrix × Vector | 9 | 6 | **15** |
| Quat × Vector | 24 | 17 | **41** |

> **Insight:** Quaternions are ~40% more efficient for composing rotations.
> Matrices are ~3× more efficient for single vector rotations.

---

## Part 4 (Extra 2pts): Angular Velocity Relation

**Claim:** For unit angular velocity about unit axis $\vec{\omega}$:
$$\dot{Q} \cdot Q^* = \left(0, \frac{\vec{\omega}}{2}\right)$$

### Setup

Angle at time $t$: $\theta(t) = t$ (unit angular velocity)

$$Q(t) = \left(\cos\frac{t}{2}, \ \sin\frac{t}{2}\vec{\omega}\right)$$

### Differentiate

$$\dot{Q} = \left(-\frac{1}{2}\sin\frac{t}{2}, \ \frac{1}{2}\cos\frac{t}{2}\vec{\omega}\right)$$

### Compute $\dot{Q}Q^*$

With $Q^* = (\cos\frac{t}{2}, -\sin\frac{t}{2}\vec{\omega})$:

**Scalar:**
$$-\frac{1}{2}\sin\frac{t}{2}\cos\frac{t}{2} + \frac{1}{2}\sin\frac{t}{2}\cos\frac{t}{2} = 0 \quad \checkmark$$

**Vector:**
$$\frac{1}{2}\sin^2\frac{t}{2}\vec{\omega} + \frac{1}{2}\cos^2\frac{t}{2}\vec{\omega} = \frac{1}{2}(\sin^2 + \cos^2)\vec{\omega} = \frac{\vec{\omega}}{2}$$

$$\boxed{\dot{Q} \cdot Q^* = \left(0, \frac{\vec{\omega}}{2}\right)}$$

> **Physical interpretation:** The factor of $\frac{1}{2}$ arises because quaternions use half-angles — directly connected to the two-to-one covering from Part 2. ∎
