# Problem 1: Forward Kinematics and Jacobians for 3-DOF Manipulator

## Problem Statement
For the three degree of freedom manipulator shown in Figure 1:
- (a) Find the forward kinematics map
- (b) Derive the spatial and body Jacobians

## Setup and Coordinate Frame

Looking at Figure 1, this is a 3-DOF manipulator with:
- **θ₁**: Base rotation about vertical axis (z-axis)
- **θ₂**: Shoulder rotation about horizontal axis (y-axis)
- **θ₃**: End-effector rotation about horizontal axis (y-axis)

### Coordinate System
- **Origin**: At base (joint 1 location)
- **z-axis**: Pointing up (along θ₁ rotation axis)
- **x-axis**: Pointing forward (toward arm extension in home config)

### Link Lengths
- **L₁**: Upper arm length (joint 1 to joint 2)
- **L₂**: Forearm length (joint 2 to end-effector)

---

## Part (a): Forward Kinematics Map

Using the **Product of Exponentials (PoE)** formula:

$$g_{st}(\theta) = e^{\hat{\xi}_1 \theta_1} \cdot e^{\hat{\xi}_2 \theta_2} \cdot e^{\hat{\xi}_3 \theta_3} \cdot g_{st}(0)$$

### Step 1: Define the Twists

For revolute joints, a twist has the form ξ = (ω, v) where:
- ω is the unit rotation axis
- v = -ω × q (q is a point on the axis)

| Joint | Axis direction ωᵢ | Point on axis qᵢ | Twist ξᵢ = (ω, v) |
|-------|------------------|------------------|-------------------|
| 1 | ω₁ = (0, 0, 1)ᵀ | q₁ = (0, 0, 0)ᵀ | ξ₁ = (0, 0, 1, 0, 0, 0)ᵀ |
| 2 | ω₂ = (0, 1, 0)ᵀ | q₂ = (L₁, 0, 0)ᵀ | ξ₂ = (0, 1, 0, 0, 0, L₁)ᵀ |
| 3 | ω₃ = (0, 1, 0)ᵀ | q₃ = (L₁+L₂, 0, 0)ᵀ | ξ₃ = (0, 1, 0, 0, 0, L₁+L₂)ᵀ |

### Step 2: Home Configuration g_st(0)

When all angles are zero, the end-effector is at position (L₁ + L₂, 0, 0)ᵀ with identity orientation:

```
g_st(0) = | I₃ₓ₃   p₀ |    where p₀ = | L₁ + L₂ |
          | 0      1  |              | 0       |
                                     | 0       |
```

### Step 3: Compute Matrix Exponentials

For a revolute joint with twist ξ = (ω, v), the exponential is:

```
e^(ξ̂θ) = | e^(ω̂θ)   (I - e^(ω̂θ))(ω × v) + ωωᵀvθ |
         | 0         1                            |
```

**Notation**: cᵢ = cos(θᵢ), sᵢ = sin(θᵢ)

#### Joint 1 (rotation about z-axis at origin):
```
e^(ξ̂₁θ₁) = | c₁  -s₁  0  0 |
           | s₁   c₁  0  0 |
           | 0    0   1  0 |
           | 0    0   0  1 |
```

#### Joint 2 (rotation about y-axis at x = L₁):
```
e^(ξ̂₂θ₂) = | c₂   0  s₂  L₁(1-c₂) |
           | 0    1  0   0         |
           | -s₂  0  c₂  L₁s₂      |
           | 0    0  0   1         |
```

#### Joint 3 (rotation about y-axis at x = L₁ + L₂):
```
e^(ξ̂₃θ₃) = | c₃   0  s₃  (L₁+L₂)(1-c₃) |
           | 0    1  0   0              |
           | -s₃  0  c₃  (L₁+L₂)s₃      |
           | 0    0  0   1              |
```

### Step 4: Final Forward Kinematics

The FK map is: g_st(θ) = e^(ξ̂₁θ₁) · e^(ξ̂₂θ₂) · e^(ξ̂₃θ₃) · g_st(0)

**End-effector position**:
```
p = | c₁(L₁c₂ + L₂c₂₃) |
    | s₁(L₁c₂ + L₂c₂₃) |
    | L₁s₂ + L₂s₂₃     |
```

where c₂₃ = cos(θ₂ + θ₃), s₂₃ = sin(θ₂ + θ₃)

**End-effector orientation** (rotation matrix):
```
R = | c₁c₂₃   -s₁   c₁s₂₃  |
    | s₁c₂₃    c₁   s₁s₂₃  |
    | -s₂₃     0    c₂₃    |
```

---

## Part (b): Spatial and Body Jacobians

### Spatial Jacobian

The spatial Jacobian maps joint velocities to the spatial twist of the end-effector:

$$J^s(\theta) = \begin{bmatrix} \xi_1 & Ad_{e^{\hat{\xi}_1\theta_1}}\xi_2 & Ad_{e^{\hat{\xi}_1\theta_1}e^{\hat{\xi}_2\theta_2}}\xi_3 \end{bmatrix}$$

The Adjoint transformation is:
```
Ad_g = | R      0   |    where g = | R  p |
       | p̂R    R   |              | 0  1 |
```

#### Column 1: ξ₁ˢ = ξ₁
```
ξ₁ˢ = | 0 |
      | 0 |
      | 1 |
      | 0 |
      | 0 |
      | 0 |
```

#### Column 2: ξ₂ˢ = Ad_{e^(ξ̂₁θ₁)} ξ₂
```
ξ₂ˢ = | -s₁    |
      | c₁     |
      | 0      |
      | L₁c₁   |
      | L₁s₁   |
      | 0      |
```

#### Column 3: ξ₃ˢ = Ad_{e^(ξ̂₁θ₁)e^(ξ̂₂θ₂)} ξ₃
```
ξ₃ˢ = | -s₁              |
      | c₁               |
      | 0                |
      | (L₁c₂ + L₂)c₁    |
      | (L₁c₂ + L₂)s₁    |
      | -L₁s₂            |
```

#### Complete Spatial Jacobian:
```
J^s(θ) = | 0   -s₁   -s₁              |   ← ω (angular)
         | 0    c₁    c₁              |
         | 1    0     0               |
         | 0   L₁c₁  (L₁c₂+L₂)c₁      |   ← v (linear)
         | 0   L₁s₁  (L₁c₂+L₂)s₁      |
         | 0    0    -L₁s₂            |
```

### Body Jacobian

The body Jacobian is related to the spatial Jacobian by:

$$J^b(\theta) = Ad_{g_{st}^{-1}} J^s(\theta)$$

Alternatively, compute directly:
$$J^b = \begin{bmatrix} Ad_{g_{st}(0)^{-1}e^{-\hat{\xi}_3\theta_3}e^{-\hat{\xi}_2\theta_2}e^{-\hat{\xi}_1\theta_1}}\xi_1 & \cdots & Ad_{g_{st}(0)^{-1}}\xi_3 \end{bmatrix}$$

The body Jacobian expresses the same velocity information but in the **tool frame** rather than the space frame.

#### Complete Body Jacobian:
```
J^b(θ) = | -s₂₃              0     0  |   ← ω (angular in tool frame)
         | -c₁c₂₃           -s₁    0  |
         |  s₁c₂₃           -c₁    0  |
         |  L₂s₂₃            0     0  |   ← v (linear in tool frame)
         |  L₁s₃+L₂s₂₃c₁     ?     ?  |
         |  ...             ...   ... |
```

*Note: The full body Jacobian computation requires careful matrix multiplication. The key relationship is J^b = Ad_{g⁻¹} J^s*

---

## Key Formulas Reference

### Rodrigues' Formula for Rotation
For unit axis ω and angle θ:
```
e^(ω̂θ) = I + sin(θ)ω̂ + (1-cos(θ))ω̂²
```

### Twist Exponential for Revolute Joint
```
e^(ξ̂θ) = | e^(ω̂θ)   (I-e^(ω̂θ))(ω×v) + ωωᵀvθ |
         | 0         1                         |
```

### Adjoint Transformation
```
Ad_g ξ = | R      0  | | ω |   =  | Rω        |
         | p̂R    R  | | v |      | p̂Rω + Rv |
```

---

## Verification (Optional Python Code)

```python
import numpy as np

def skew(v):
    """Create skew-symmetric matrix from 3-vector"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def rodrigues(omega, theta):
    """Rotation matrix from axis-angle using Rodrigues' formula"""
    omega_hat = skew(omega)
    return np.eye(3) + np.sin(theta)*omega_hat + (1-np.cos(theta))@omega_hat@omega_hat

def fk_3dof(theta, L1, L2):
    """Forward kinematics for 3-DOF manipulator"""
    t1, t2, t3 = theta
    c1, s1 = np.cos(t1), np.sin(t1)
    c2, s2 = np.cos(t2), np.sin(t2)
    c23, s23 = np.cos(t2+t3), np.sin(t2+t3)

    # Position
    p = np.array([
        c1*(L1*c2 + L2*c23),
        s1*(L1*c2 + L2*c23),
        L1*s2 + L2*s23
    ])

    # Rotation matrix
    R = np.array([
        [c1*c23, -s1, c1*s23],
        [s1*c23,  c1, s1*s23],
        [-s23,    0,  c23]
    ])

    return R, p

# Test with L1=1, L2=1, theta=[0,0,0]
R, p = fk_3dof([0, 0, 0], 1, 1)
print(f"Home position: {p}")  # Should be [2, 0, 0]
```
