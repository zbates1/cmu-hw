# Literature Review: Physics-Based and Data-Driven Modeling of Liver Vasculature

## Overview
This literature review examines six key papers on liver vascular modeling, mesh generation, and physics-based simulation. These papers span approaches from algorithmic tree generation to advanced multiphysics coupling, providing a comprehensive foundation for implementing vascular mesh generation algorithms.

---

## 1. A mesh-based model of liver vasculature: Implications for improved radiation dosimetry (Correa-Alfonso et al., 2022)

### Problem Statement
Accurate radiation dosimetry in liver radiopharmaceutical treatments requires realistic 3D models of hepatic vasculature including arterial (HA), portal venous (HPV), and hepatic venous (HV) trees.

### Methodology
- **Algorithmic tree growth**: Trees are grown iteratively inside reference adult liver lobe volumes using straight cylindrical segments
- **Hemodynamic constraints**: Each iteration updates geometry and hemodynamic parameters (flow rates, pressures, vessel radii)
- **Vessel representation**: Vessels modeled as straight cylinders with radius determined by Murray's law or similar scaling
- **Mesh generation**: Final volume is tetrahedralized to include vessels for coupling with radiation transport simulations

### Key Contributions
- Computer-generated vascular tree models embedded in anatomically realistic liver geometry
- Integration of vascular trees with tetrahedral meshing for physics simulation
- Practical application to radiation dosimetry

### Relevance to Project
**High relevance** - This paper directly addresses tree generation + meshing, which is ideal for the programming option. The algorithmic approach is implementable and testable.

### Strengths
- Practical, implementable algorithm
- Combines geometric modeling with physics-based constraints
- Clear application domain

### Limitations
- Simplified vessel geometry (straight cylinders)
- Limited validation against actual patient data
- Does not model vessel branching morphology in detail

---

## 2. Rigorous mathematical optimization of synthetic hepatic vascular trees (Jessen et al., 2022)

### Problem Statement
Previous vascular tree generation methods rely on heuristics and growth rules. This paper seeks a mathematically rigorous optimization framework to generate synthetic trees matching real liver vasculature.

### Methodology
- **Optimization formulation**: Nonlinear optimization problem with objective functions based on:
  - Murray's law (minimizing pumping power + blood volume)
  - Morphological constraints (branching angles, symmetry ratios)
  - Space-filling properties
- **Validation**: Synthetic trees compared against human liver corrosion cast data
- **Metrics**: Structural metrics include diameter ratios, branching angles, generation counts, space-filling indices

### Key Mathematical Framework
The optimization seeks tree topology and geometry (node positions, radii) that minimize:

**Cost function**: E = α·E_hemodynamic + β·E_morphology + γ·E_space-filling

Where:
- E_hemodynamic: Based on Murray's law (r³ = r₁³ + r₂³)
- E_morphology: Penalizes unrealistic branching angles and asymmetry
- E_space-filling: Encourages perfusion coverage

### Key Contributions
- First rigorous optimization framework for synthetic liver trees
- Validation against real anatomical data
- High-quality synthetic trees for computational studies

### Relevance to Project
**Medium-high relevance** - More complex than simple algorithmic growth, but provides theoretical foundation. Implementation would be challenging but rewarding.

### Strengths
- Mathematically rigorous
- Validated against real data
- Produces high-quality synthetic trees

### Limitations
- Computationally expensive optimization
- Requires careful tuning of weights (α, β, γ)
- Implementation complexity is high

---

## 3. Computational Modeling of the Liver Arterial Blood Flow for Microsphere Therapy (Taebi et al., 2020)

### Problem Statement
Understanding how boundary conditions affect blood flow and microsphere distribution in hepatic arterial trees for cancer therapy planning.

### Methodology
- **Image-based geometry**: Hepatic arterial tree extracted from medical imaging
- **CFD simulation**: Computational fluid dynamics with Navier-Stokes equations
- **Boundary condition study**: Investigates impact of different outlet boundary conditions on flow patterns
- **Particle tracking**: Microsphere transport and distribution analysis

### Key Physics
- **Navier-Stokes equations** for incompressible flow:
  - ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u
- **Boundary conditions**:
  - Inlet: Pulsatile flow waveform
  - Outlets: Zero pressure, resistance boundary conditions, or impedance

### Key Contributions
- Demonstrates importance of boundary conditions in liver hemodynamics
- Relevant for therapy planning
- Bridges geometric modeling and physics simulation

### Relevance to Project
**Medium relevance** - Focuses on CFD rather than mesh generation. Could be used to demonstrate physics-based analysis *after* generating meshes.

### Strengths
- Clinical relevance
- Thorough CFD methodology
- Sensitivity analysis of boundary conditions

### Limitations
- Requires medical imaging data (not synthetic generation)
- CFD implementation is complex
- Focus is on flow physics rather than geometry generation

---

## 4. Connecting continuum poroelasticity with discrete synthetic vascular trees (Ebrahem et al., 2023)

### Problem Statement
Liver tissue behavior involves both discrete vascular flow and continuum tissue mechanics. How can we couple these two scales effectively?

### Methodology
- **Discrete vascular tree**: Synthetic tree generation using algorithmic/optimization methods
- **Continuum poroelasticity**: Liver parenchyma modeled as poroelastic medium (solid matrix + fluid)
- **Multiscale coupling**:
  - 1D flow in vessels (Poiseuille flow)
  - 3D Darcy flow in tissue
  - Elastic deformation of tissue
- **Mathematical framework**: Biot's poroelasticity equations coupled with discrete tree flow

### Key Physics & Mathematics

**1D vessel flow (Hagen-Poiseuille)**:
```
Q = (π r⁴ / 8μ) · (ΔP / L)
```

**3D Darcy flow in tissue**:
```
q = -(k/μ) ∇p
where k is permeability
```

**Poroelastic coupling**:
```
∇·σ' - α∇p = 0  (momentum balance)
∂(ζ/M)/∂t + α∂ε_v/∂t - ∇·q = 0  (mass balance)
where σ' is effective stress, α is Biot coefficient, ζ is fluid pressure
```

### Key Contributions
- Multiscale framework coupling discrete vessels and continuum tissue
- Integrates perfusion and mechanical deformation
- Advanced multiphysics approach

### Relevance to Project
**High relevance but high complexity** - Shows state-of-the-art multiphysics. Could implement simplified version: tree generation + simple perfusion model.

### Strengths
- Comprehensive multiphysics framework
- Rigorous mathematical formulation
- Addresses multiple length scales

### Limitations
- Implementation is very complex
- Requires expertise in finite element methods
- May be too ambitious for project timeline

---

## 5. Modeling of the contrast-enhanced perfusion test in liver (Rohan et al., 2016)

### Problem Statement
Model contrast agent perfusion in liver accounting for both large vessels (1D flow) and tissue capillary beds (3D porous flow).

### Methodology
- **Multi-compartment model**:
  - Major vessels: 1D flow networks
  - Parenchyma: 3D porous medium with dual porosity (arterial + venous compartments)
- **Homogenization theory**: Derive effective continuum properties from microstructure
- **Contrast transport**: Advection-diffusion of contrast agent through vascular network and tissue

### Key Mathematics

**1D vessel network flow**:
```
∂(A)/∂t + ∂(Q)/∂x = 0  (mass conservation)
∂Q/∂t + ∂(Q²/A)/∂x + (A/ρ)∂P/∂x = -f·Q  (momentum)
```

**3D porous medium flow**:
```
∂p_a/∂t - ∇·(K_a ∇p_a) = S_av (arterial compartment)
∂p_v/∂t - ∇·(K_v ∇p_v) = -S_av (venous compartment)
S_av = transfer between arterial and venous
```

### Key Contributions
- Multiscale perfusion model
- Application to contrast imaging simulation
- Theoretical framework using homogenization

### Relevance to Project
**Medium relevance** - Alternative to tree generation focusing on perfusion modeling. Could complement geometric models.

### Strengths
- Strong theoretical foundation
- Clinically motivated (contrast imaging)
- Multiscale approach

### Limitations
- Heavy on continuum modeling, less on discrete geometry
- Homogenization theory is mathematically advanced
- May not be ideal for "mesh generation" focus

---

## 6. A multiscale-multiphysics framework for modeling organ-scale liver regrowth (Ebrahem et al., 2025)

### Problem Statement
Model liver regeneration after partial hepatectomy, including vascular remodeling, perfusion-driven growth, and mechanical deformation.

### Methodology
- **Synthetic vascular tree generation**: Algorithmic generation of HA, PV, HV trees
- **Homogenized perfusion model**: Continuum representation of blood flow in tissue
- **Poroelastic growth**: Tissue growth driven by perfusion and mechanical stimuli
- **Growth law**: Tissue mass increases based on nutrient/oxygen delivery
- **Temporal evolution**: Simulation of liver regrowth over weeks post-resection

### Key Physics

**Growth law (simplified)**:
```
∂m/∂t = f(perfusion, stress)
```
Typically:
```
∂m/∂t = k_growth · (p - p_threshold)⁺
where p is perfusion/pressure
```

**Multiphysics coupling**:
1. Vascular tree provides perfusion sources
2. Darcy/Brinkman flow distributes blood in tissue
3. Perfusion drives growth
4. Growth causes mechanical deformation
5. Deformation affects vascular geometry → feedback loop

### Key Contributions
- State-of-the-art multiscale-multiphysics model
- Application to liver regeneration (clinical relevance)
- Integrates multiple biological processes

### Relevance to Project
**Medium relevance** - Very advanced, but good for background/motivation. Could cite as inspiration for future work.

### Strengths
- Comprehensive biological model
- Cutting-edge research (2025 publication)
- High clinical impact

### Limitations
- Extremely complex to implement
- Requires significant computational resources
- Beyond scope of typical course project

---

## Comparative Analysis

| Paper | Focus | Complexity | Implementation Feasibility | Best For |
|-------|-------|------------|---------------------------|----------|
| Correa-Alfonso 2022 | Tree generation + meshing | Low-Medium | **High** | **Primary algorithm choice** |
| Jessen 2022 | Optimization-based trees | Medium-High | Medium | Alternative/advanced approach |
| Taebi 2020 | CFD hemodynamics | Medium | Medium | Post-processing/validation |
| Ebrahem 2023 | Multiscale perfusion-mechanics | High | Low | Background/theory |
| Rohan 2016 | Multi-compartment perfusion | Medium-High | Low-Medium | Alternative focus |
| Ebrahem 2025 | Growth-perfusion-mechanics | Very High | Very Low | Motivation/future work |

---

## Recommended Implementation Strategy

Based on this review, I recommend the following approach for your programming project:

### Primary Algorithm: Vascular Tree Generation (Correa-Alfonso 2022)
**Why**: Implementable, directly relevant, combines geometry + meshing

**Core Steps**:
1. Define liver volume (simple ellipsoid or from data)
2. Generate synthetic vascular trees using constrained growth algorithm
3. Apply Murray's law for vessel radii
4. Tetrahedralize the domain including vessels

### Secondary Enhancement: Optimization-based refinement (Jessen 2022)
**Why**: Adds rigor, improves tree quality

**Core Steps**:
1. Start with algorithmic tree
2. Optimize node positions and radii to minimize cost function
3. Validate against morphological metrics

### Validation/Demonstration: Simple perfusion analysis (Taebi 2020 / Ebrahem 2023)
**Why**: Shows "physics-based" aspect, demonstrates mesh utility

**Core Steps**:
1. Compute flow distribution using Poiseuille law
2. Visualize pressure/flow fields
3. Simple sensitivity analysis

---

## Key Formulas to Implement

### 1. Murray's Law (Vessel Radii)
At each bifurcation:
```
r_parent³ = r_child1³ + r_child2³
```

### 2. Hagen-Poiseuille Flow (1D vessels)
```
Q = (π r⁴ / 8μ) · (ΔP / L)
Q: flow rate
r: vessel radius
μ: blood viscosity (~3-4 cP = 0.003-0.004 Pa·s)
ΔP: pressure drop
L: vessel length
```

### 3. Resistance of vessel segment
```
R = 8μL / (π r⁴)
```

### 4. Darcy's Law (Tissue perfusion)
```
q = -(k/μ) ∇p
q: flux
k: permeability
p: pressure
```

### 5. Space-filling metric
```
D_box = lim (log N(ε) / log(1/ε))
N(ε): number of boxes of size ε covering the tree
Fractal dimension indicator
```

---

## Gaps and Future Directions

1. **Patient-specific modeling**: Move from synthetic to image-based trees
2. **Fluid-structure interaction**: Vessel wall compliance and deformation
3. **Pathological modeling**: Tumors, cirrhosis, stenosis effects
4. **Machine learning integration**: Learn tree generation from data
5. **Multi-organ coupling**: Liver + systemic circulation
6. **Temporal dynamics**: Pulsatile flow, growth, remodeling

---

## Conclusion

The reviewed papers provide a solid foundation for implementing a liver vascular mesh generation project. The Correa-Alfonso (2022) approach offers the most practical starting point, with opportunities to incorporate optimization (Jessen 2022) and physics-based validation (Taebi 2020). More advanced multiphysics frameworks (Ebrahem 2023, 2025) provide theoretical depth and motivation for future extensions.

**Recommended focus**: Implement algorithmic vascular tree generation with Murray's law constraints, followed by tetrahedral meshing, and demonstrate with simple hemodynamic analysis.
