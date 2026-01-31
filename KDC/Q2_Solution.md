# Question 2: Planar Rigid Body Transformations

> **Notation:**
> - $g = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix} \in SE(2)$, where $R \in SO(2)$, $p \in \mathbb{R}^2$
> - Twist: $\hat{\xi} = \begin{bmatrix} \hat{\omega} & v \\ 0 & 0 \end{bmatrix}$ with $\hat{\omega} = \begin{bmatrix} 0 & -\omega \\ \omega & 0 \end{bmatrix}$
> - Twist coordinates: $\xi = (v_x, v_y, \omega)^T \in \mathbb{R}^3$

---

## Part 1 (2pts): Show exp(ξ̂) ∈ SE(2)

### Case 1: Pure Translation (ω = 0)

$$\hat{\xi}^2 = 0 \implies \exp(\hat{\xi}) = I + \hat{\xi} = \begin{bmatrix} I & v \\ 0 & 1 \end{bmatrix} \in SE(2) \quad \checkmark$$

### Case 2: General Case (ω ≠ 0)

**Rotation block:** $e^{\hat{\omega}} = \begin{bmatrix} \cos\omega & -\sin\omega \\ \sin\omega & \cos\omega \end{bmatrix} \in SO(2)$

**Translation block:** $p = Av$ where $A = \frac{1}{\omega}\begin{bmatrix} \sin\omega & -(1-\cos\omega) \\ 1-\cos\omega & \sin\omega \end{bmatrix}$

$$\exp(\hat{\xi}) = \begin{bmatrix} R(\omega) & Av \\ 0 & 1 \end{bmatrix} \in SE(2) \quad \checkmark$$

---

## Part 2 (1pt): Twist Coordinates for Pure Motions

### Pure Rotation about q = (qₓ, qᵧ) with ω = 1

Velocity at origin from rotation about q:

$$v = \hat{\omega}(0 - q) = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} -q_x \\ -q_y \end{bmatrix} = \begin{bmatrix} q_y \\ -q_x \end{bmatrix}$$

$$\boxed{\xi = \begin{bmatrix} q_y \\ -q_x \\ 1 \end{bmatrix}}$$

### Pure Translation with velocity (vₓ, vᵧ)

$$\boxed{\xi = \begin{bmatrix} v_x \\ v_y \\ 0 \end{bmatrix}}$$

---

## Part 3 (2pts): Every Planar Motion is Rotation or Translation

Given twist $\xi = (v_x, v_y, \omega)^T$:

### Case 1: ω = 0
Pure translation by $(v_x, v_y)$. ✓

### Case 2: ω ≠ 0

Find **pole** q where velocity = 0. Solve $v + \hat{\omega}q = 0$:

$$\begin{bmatrix} 0 & -\omega \\ \omega & 0 \end{bmatrix}\begin{bmatrix} q_x \\ q_y \end{bmatrix} = \begin{bmatrix} -v_x \\ -v_y \end{bmatrix} \implies \boxed{q = \begin{pmatrix} -v_y/\omega \\ v_x/\omega \end{pmatrix}}$$

Motion is pure rotation about this unique pole. ✓

> *This is Chasles' theorem in 2D.*

---

## Part 4 (2pts): Spatial and Body Twists

With $g = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix}$, $g^{-1} = \begin{bmatrix} R^T & -R^Tp \\ 0 & 1 \end{bmatrix}$:

### V̂ˢ = ġg⁻¹ is a twist

$$\dot{g}g^{-1} = \begin{bmatrix} \dot{R}R^T & \dot{p} - \dot{R}R^Tp \\ 0 & 0 \end{bmatrix}$$

Since $RR^T = I \Rightarrow \dot{R}R^T$ is **skew-symmetric** $\Rightarrow \hat{V}^s \in se(2)$ ✓

### V̂ᵇ = g⁻¹ġ is a twist

$$g^{-1}\dot{g} = \begin{bmatrix} R^T\dot{R} & R^T\dot{p} \\ 0 & 0 \end{bmatrix}$$

$R^T\dot{R}$ is skew-symmetric $\Rightarrow \hat{V}^b \in se(2)$ ✓

### Interpretation

| Twist | Frame | Meaning |
|-------|-------|---------|
| **Vˢ** (spatial) | Fixed world | Velocity seen by stationary observer |
| **Vᵇ** (body) | Moving body | Velocity felt by observer on body |

```
   Fixed {s}                    Body {b}
      ↑ y                          ↑ y'
      |    ●→ Vˢ                   |    ●→ Vᵇ
      |   /                        |   /
      |__/____→ x                  |__/____→ x'
        body                      (moves with body)

   "World sees this"           "Body feels this"
```

**Relation:** $V^s = \text{Ad}_g \cdot V^b$

---

## Part 5 (1pt): Adjoint Transformation

From $\hat{V}^s = g\hat{V}^bg^{-1}$:

$$g\hat{V}^bg^{-1} = \begin{bmatrix} R\hat{\omega}R^T & Rv^b - R\hat{\omega}R^Tp \\ 0 & 0 \end{bmatrix}$$

In 2D: $R\hat{\omega}R^T = \hat{\omega}$ (same ω), and $\hat{\omega}p = \omega[-p_y, p_x]^T$

So: $\omega^s = \omega^b$ and $v^s = Rv^b + \omega[p_y, -p_x]^T$

$$\boxed{\text{Ad}_g = \begin{bmatrix} R & \begin{bmatrix} p_y \\ -p_x \end{bmatrix} \\[4pt] 0 & 1 \end{bmatrix}}$$

> *The $[p_y, -p_x]^T$ term: rotation about a displaced point creates additional linear velocity (lever arm effect).*

---

## Summary

| Part | Result | Equations |
|------|--------|:---------:|
| 1 | $\exp(\hat{\xi}) \in SE(2)$ via nilpotent (ω=0) or Rodrigues (ω≠0) | 3 |
| 2 | Rotation: $\xi = [q_y, -q_x, 1]^T$; Translation: $\xi = [v_x, v_y, 0]^T$ | 2 |
| 3 | Pole at $q = (-v_y/\omega, v_x/\omega)$ or pure translation | 2 |
| 4 | $\hat{V}^s = \dot{g}g^{-1}$, $\hat{V}^b = g^{-1}\dot{g}$ both in $se(2)$ | 2 |
| 5 | $\text{Ad}_g = [R, [p_y,-p_x]^T; 0, 1]$ | 2 |

> **Reference:** Murray, Li, Sastry, *A Mathematical Introduction to Robotic Manipulation*, Ch. 2
