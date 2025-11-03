# Implementation Guide: Liver Vascular Mesh Generation

## Project Scope

This guide outlines the mathematical formulas and algorithms for implementing a liver vascular mesh generation system suitable for a graduate-level computational modeling course project.

**Target deliverables**:
1. Vascular tree generation algorithm (Python)
2. Tetrahedral mesh generation with embedded vessels (Python)
3. Simple hemodynamic analysis demonstration
4. 8-12 page report with results

---

## Part 1: Mathematical Foundation

### 1.1 Murray's Law (Optimal Vessel Radii)

Murray's law states that at a vessel bifurcation, the cube of the parent radius equals the sum of cubes of child radii, minimizing the energy required for blood flow.

**Formula**:
```
r₀³ = r₁³ + r₂³

where:
  r₀ = parent vessel radius
  r₁, r₂ = child vessel radii
```

**Derivation (simplified)**:
The total power required for blood flow consists of:
1. **Metabolic cost** of maintaining blood volume: P_metabolic ∝ V ∝ r²L
2. **Pumping cost** to overcome viscous resistance: P_pump ∝ Q²R ∝ Q²L/r⁴

Total power: P_total = k₁r²L + k₂Q²L/r⁴

Minimizing with respect to r (for bifurcation with flow conservation Q₀ = Q₁ + Q₂):
→ r³ ∝ Q

Therefore: r₀³ = r₁³ + r₂³

**Implementation note**: For asymmetric bifurcations, define a bifurcation ratio:
```python
beta = r₁ / r₂  # typically 1.0 to 2.0
# Then solve:
# r₀³ = r₁³ + r₂³
# r₁ = beta * r₂
```

### 1.2 Hagen-Poiseuille Flow (1D Vessel Flow)

For laminar flow in a cylindrical tube:

**Formula**:
```
Q = (π r⁴ ΔP) / (8 μ L)

where:
  Q = volumetric flow rate (m³/s)
  r = vessel radius (m)
  ΔP = pressure drop along vessel (Pa)
  μ = dynamic viscosity (Pa·s)
  L = vessel length (m)
```

**Resistance formulation**:
```
R = ΔP / Q = (8 μ L) / (π r⁴)

ΔP = Q · R  (analogous to Ohm's law: V = I·R)
```

**Typical values for blood**:
- μ (blood viscosity): 3-4 cP = 0.003-0.004 Pa·s
- For large vessels: use 0.0035 Pa·s
- For small vessels: may need to account for Fåhræus-Lindqvist effect (viscosity decreases in small vessels)

**Pressure-flow network equations**:
For a tree network, at each node (junction):
1. **Flow conservation (Kirchhoff's current law)**:
   ```
   Σ Q_in = Σ Q_out
   ```

2. **Pressure-flow relationship (Poiseuille)**:
   ```
   Q_i = (P_parent - P_child) / R_i
   ```

This creates a system of linear equations solvable for pressures at all nodes.

### 1.3 Vessel Branching Geometry

**Bifurcation angles**:
Optimal branching angles minimize energy (Murray 1926, Zamir 1976):

```
cos(θ₁) = (r₀⁴ + r₁⁴ - r₂⁴) / (2 r₀² r₁²)
cos(θ₂) = (r₀⁴ + r₂⁴ - r₁⁴) / (2 r₀² r₂²)

where:
  θ₁ = angle between parent and child 1
  θ₂ = angle between parent and child 2
```

**Simplified version** (for equal children r₁ = r₂):
```
θ₁ = θ₂ ≈ 37.5° (from parent axis)
Included angle ≈ 75°
```

**Practical implementation**:
For computational simplicity, you can use:
- Symmetric bifurcation: both children at ±30-40° from parent
- Asymmetric bifurcation: larger child at ~20°, smaller at ~50°

### 1.4 Space-Filling Tree Growth

Trees should efficiently perfuse the liver volume. Key principles:

**Terminal vessel spacing**:
```
d_terminal ≈ 2 * r_capillary_field

Typical values:
  d_terminal ≈ 200-500 μm (0.2-0.5 mm)
```

**Perfusion volume per terminal**:
```
V_perfusion = (4/3) π R_perfusion³

where R_perfusion ≈ 2-3 mm for liver
```

**Number of terminals needed**:
```
N_terminals = V_liver / V_perfusion

Example:
  V_liver ≈ 1500 cm³
  V_perfusion ≈ 33.5 mm³ (R=2mm)
  N_terminals ≈ 45,000
```

For a simplified project, use N_terminals ~ 100-500.

---

## Part 2: Algorithms for Implementation

### Algorithm 1: Constrained Constructive Optimization (CCO)

**Reference**: Schreiner & Buxbaum (1993), adapted by Karch et al.

**Concept**: Grow tree iteratively by adding terminal segments that minimize a cost function.

**Pseudocode**:
```
function GrowVascularTree(root_point, root_radius, N_terminals, domain):
    tree = initialize_tree(root_point, root_radius)

    for i = 1 to N_terminals:
        # 1. Generate candidate terminal locations
        candidates = generate_random_points_in_domain(domain, N_candidates=100)

        # 2. For each candidate, find optimal connection point
        best_cost = infinity
        best_candidate = None
        best_connection = None

        for candidate in candidates:
            # Try connecting to each existing segment
            for segment in tree.segments:
                connection_point = find_optimal_connection(segment, candidate)
                new_tree = tree.add_bifurcation(segment, connection_point, candidate)

                # Update radii using Murray's law
                new_tree.update_radii_murrays_law()

                # Compute cost function
                cost = compute_tree_cost(new_tree)

                if cost < best_cost:
                    best_cost = cost
                    best_candidate = candidate
                    best_connection = connection_point

        # 3. Add best terminal to tree
        tree.add_bifurcation(best_segment, best_connection, best_candidate)
        tree.update_radii_murrays_law()

    return tree

function compute_tree_cost(tree):
    # Volume cost: sum of vessel volumes
    V_total = sum(π * r² * L for each segment)

    # Pumping cost: sum of flow resistances
    R_total = sum(8*μ*L / (π*r⁴) for each segment)

    # Weighted cost
    cost = α * V_total + β * R_total

    return cost
```

**Simplifications for project**:
1. Use 2D initially, then extend to 3D
2. Reduce N_terminals to 50-200
3. Use fewer candidate points (10-20 instead of 100)
4. Skip optimization step (just use closest existing segment)

### Algorithm 2: Lindenmayer System (L-System) Growth

**Reference**: Prusinkiewicz & Lindenmayer (1990)

**Concept**: Use grammar-based rules to grow fractal-like trees.

**Basic L-System for binary tree**:
```
Axiom: A
Rules:
  A → B[+A][-A]
  B → BB

where:
  A = growing tip (apex)
  B = vessel segment
  [, ] = push/pop position stack
  + = rotate +θ
  - = rotate -θ
```

**Enhanced rules with radius**:
```
Axiom: A(r₀)
Rules:
  A(r) → B(r)[+A(r*k)][-A(r*k)]
  where k = (0.5)^(1/3) ≈ 0.794  (from Murray's law)
```

**Pseudocode**:
```python
def generate_tree_lsystem(axiom, rules, generations):
    current = axiom
    for gen in range(generations):
        next_string = ""
        for symbol in current:
            if symbol in rules:
                next_string += rules[symbol]
            else:
                next_string += symbol
        current = next_string
    return current

def interpret_lsystem(lstring, angle, initial_length, initial_radius):
    position = [0, 0, 0]
    direction = [0, 0, 1]  # upward initially
    radius = initial_radius
    length = initial_length

    stack = []
    segments = []

    for symbol in lstring:
        if symbol == 'B':
            new_pos = position + direction * length
            segments.append(Segment(position, new_pos, radius))
            position = new_pos
        elif symbol == 'A':
            # Terminal node
            pass
        elif symbol == '[':
            stack.append((position, direction, radius, length))
        elif symbol == ']':
            position, direction, radius, length = stack.pop()
        elif symbol == '+':
            direction = rotate(direction, +angle)
        elif symbol == '-':
            direction = rotate(direction, -angle)
        # Add more symbols for 3D rotations

    return segments
```

**Advantages**:
- Very fast generation
- Deterministic and reproducible
- Beautiful fractal structures

**Disadvantages**:
- Less control over space-filling
- May not respect anatomical constraints
- Less realistic than CCO

### Algorithm 3: Simplified Hybrid Approach (Recommended for Project)

Combine simplicity of L-system with space-filling of CCO:

**Pseudocode**:
```
function GenerateSimplifiedTree(root_pos, root_radius, max_generations, domain):
    tree = [Segment(root_pos, root_pos + [0,0,10], root_radius)]
    queue = [(tree[0], generation=0)]

    while queue is not empty:
        parent_seg, gen = queue.pop()

        if gen >= max_generations:
            continue

        # Get parent endpoint and direction
        pos = parent_seg.end_point
        dir = parent_seg.direction

        # Generate two children with Murray's law
        r0 = parent_seg.radius
        r1 = r2 = r0 * (0.5)^(1/3)  # symmetric bifurcation

        # Child directions (branching angles)
        theta = 35 * π/180  # 35 degrees from parent
        dir1 = rotate(dir, +theta, random_axis)
        dir2 = rotate(dir, -theta, random_axis)

        # Child lengths (decrease with generation)
        length = parent_seg.length * 0.8

        # Check if children are inside domain
        end1 = pos + dir1 * length
        end2 = pos + dir2 * length

        if inside_domain(end1, domain):
            seg1 = Segment(pos, end1, r1)
            tree.append(seg1)
            queue.append((seg1, gen+1))

        if inside_domain(end2, domain):
            seg2 = Segment(pos, end2, r2)
            tree.append(seg2)
            queue.append((seg2, gen+1))

    return tree
```

**This approach**:
- Simple to implement
- Fast execution
- Produces reasonable trees
- Easy to modify and extend

---

## Part 3: Tetrahedral Mesh Generation

### 3.1 Mesh Requirements

For finite element analysis with embedded vessels:

1. **Vessel refinement**: Finer mesh near vessels
2. **Quality constraints**: No sliver elements (aspect ratio < 10)
3. **Size field**: Gradual transition from fine to coarse
4. **Vessel representation**: Vessels as 1D elements or embedded surfaces

### 3.2 Using Existing Libraries

**Recommended: Use existing tetrahedral mesh generators**

**Option 1: TetGen** (http://wias-berlin.de/software/tetgen/)
- Robust Delaunay tetrahedralization
- Quality guarantees
- Constrained meshing (can enforce vessel boundaries)

**Python interface**:
```python
import tetgen
import numpy as np

# Define points and facets
points = np.array([...])  # vessel tree points
facets = np.array([...])  # vessel surface triangles

# Generate mesh
tet = tetgen.TetGen(points, facets)
tet.tetrahedralize(order=1, mindihedral=10, quality=True)

nodes = tet.node
elements = tet.elem
```

**Option 2: pygalmesh** (Python binding for CGAL)
- Advanced mesh generation
- Size fields
- Feature preservation

```python
import pygalmesh
import meshio

# Define domain
class LiverDomain:
    def __init__(self, vessels):
        self.vessels = vessels

    def eval(self, x):
        # Return negative inside domain, positive outside
        dist_to_boundary = ...
        return dist_to_boundary

# Generate mesh
mesh = pygalmesh.generate_mesh(
    LiverDomain(vessels),
    max_cell_circumradius=2.0,
    max_edge_size_at_feature_edges=0.5
)

meshio.write("liver_mesh.vtk", mesh)
```

**Option 3: meshio + meshpy** (Pure Python, simpler but less powerful)
```python
from meshpy.tet import MeshInfo, build

mesh_info = MeshInfo()
mesh_info.set_points(points)
mesh_info.set_facets(facets)

mesh = build(mesh_info, max_volume=1.0)

nodes = np.array(mesh.points)
elements = np.array(mesh.elements)
```

### 3.3 Vessel Mesh Representation

**Approach 1: 1D line elements** (simplest)
- Represent vessels as 1D beam/pipe elements
- Embed in 3D tetrahedral mesh
- Couple via source terms

**Approach 2: 3D embedded surfaces**
- Represent vessel walls as cylindrical surfaces
- Mesh interior and exterior separately
- More realistic but more complex

**For this project, recommend Approach 1** (1D elements).

---

## Part 4: Hemodynamic Analysis

### 4.1 Network Flow Calculation

Given a tree with N nodes, compute pressure and flow in all segments.

**Input**:
- Tree topology (parent-child relationships)
- Segment radii, lengths
- Boundary conditions: inlet pressure P_inlet, outlet pressures P_outlet

**Output**:
- Pressure at each node
- Flow rate in each segment

**Method**: Solve linear system

**Setup**:
1. Number all nodes: root = 0, terminals = N-1, N-2, ..., N-M
2. Number all segments: 0 to S-1

**Equations**:
For each internal node j (not root or terminal):
```
Σ_i Q_i = 0  (flow conservation)

where Q_i = (P_parent(i) - P_j) / R_i for incoming segments
      Q_i = (P_j - P_child(i)) / R_i for outgoing segments
```

**Matrix formulation**:
```
A * P = b

where:
  A_ij encodes flow conservation and resistances
  P = [P_1, P_2, ..., P_N]^T (unknown pressures)
  b encodes boundary conditions
```

**Python implementation**:
```python
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def compute_hemodynamics(tree, P_inlet, P_outlet, mu=0.0035):
    N = tree.num_nodes()
    A = lil_matrix((N, N))
    b = np.zeros(N)

    # Root node: fixed pressure
    A[0, 0] = 1.0
    b[0] = P_inlet

    # Terminal nodes: fixed pressure
    for term in tree.terminals:
        A[term.id, term.id] = 1.0
        b[term.id] = P_outlet

    # Internal nodes: flow conservation
    for node in tree.internal_nodes:
        for seg in node.incoming_segments:
            parent_id = seg.parent_node.id
            R = seg.resistance(mu)
            A[node.id, parent_id] += 1.0 / R
            A[node.id, node.id] -= 1.0 / R

        for seg in node.outgoing_segments:
            child_id = seg.child_node.id
            R = seg.resistance(mu)
            A[node.id, child_id] += 1.0 / R
            A[node.id, node.id] -= 1.0 / R

    # Solve
    P = spsolve(A.tocsr(), b)

    # Compute flows
    for seg in tree.segments:
        dP = P[seg.parent_node.id] - P[seg.child_node.id]
        seg.flow = dP / seg.resistance(mu)

    return P, tree
```

### 4.2 Visualization Metrics

**Useful quantities to visualize**:

1. **Pressure distribution**:
   ```python
   pressure_drop = P_inlet - P_terminals.mean()
   ```

2. **Flow distribution**:
   ```python
   total_flow = sum(Q_terminals)
   flow_uniformity = std(Q_terminals) / mean(Q_terminals)
   ```

3. **Shear stress** (on vessel walls):
   ```
   τ_wall = (4 μ Q) / (π r³)
   ```

4. **Reynolds number** (check laminar assumption):
   ```
   Re = (ρ v d) / μ = (4 ρ Q) / (π d μ)

   Typically Re < 2300 for laminar flow
   For blood in large vessels: Re ~ 100-1000
   ```

---

## Part 5: Implementation Roadmap

### Phase 1: Basic Tree Generation (Week 1)
- [ ] Implement Segment and Node classes
- [ ] Implement simplified hybrid tree generation (Algorithm 3)
- [ ] Generate tree in 3D with Murray's law
- [ ] Visualize tree (matplotlib or mayavi)
- [ ] **Deliverable**: Python script generating tree, 3D plot

### Phase 2: Mesh Generation (Week 2)
- [ ] Define liver domain (ellipsoid or simplified shape)
- [ ] Extract vessel points and connectivity
- [ ] Generate tetrahedral mesh using TetGen or pygalmesh
- [ ] Visualize mesh with embedded vessels
- [ ] **Deliverable**: Python script generating mesh, VTK output

### Phase 3: Hemodynamic Analysis (Week 3)
- [ ] Implement resistance calculation for segments
- [ ] Build linear system for network flow
- [ ] Solve for pressures and flows
- [ ] Compute derived quantities (shear stress, Re)
- [ ] Visualize results (color-coded vessels by pressure/flow)
- [ ] **Deliverable**: Python script with analysis, plots

### Phase 4: Testing and Validation (Week 4)
- [ ] Test with different tree parameters (N_generations, angles, etc.)
- [ ] Compare results to literature values (pressures, flows)
- [ ] Sensitivity analysis
- [ ] Generate figures for report
- [ ] **Deliverable**: Results, tables, figures

### Phase 5: Report Writing (Week 4-5)
- [ ] Introduction & motivation
- [ ] Methods (algorithms, formulas)
- [ ] Results
- [ ] Discussion
- [ ] Conclusion
- [ ] **Deliverable**: 8-12 page report

---

## Part 6: Key Formulas Summary

| Formula | Symbol Definitions | Typical Values |
|---------|-------------------|----------------|
| **Murray's Law** | r₀³ = r₁³ + r₂³ | r₀: parent radius, r₁,r₂: child radii |
| **Poiseuille Flow** | Q = (πr⁴ΔP)/(8μL) | μ = 0.0035 Pa·s, Q in m³/s |
| **Resistance** | R = 8μL/(πr⁴) | R in Pa·s/m³ |
| **Shear Stress** | τ = 4μQ/(πr³) | τ ~ 1-7 Pa for arteries |
| **Reynolds Number** | Re = 4ρQ/(πdμ) | ρ ≈ 1060 kg/m³ (blood), Re < 2300 for laminar |
| **Bifurcation Angle** | cos(θ) = (r₀⁴+r₁⁴-r₂⁴)/(2r₀²r₁²) | θ ~ 30-50° typically |
| **Flow Conservation** | Σ Q_in = Σ Q_out | At each node |

---

## Part 7: Expected Challenges & Solutions

### Challenge 1: Tree growing outside domain
**Solution**:
- Add collision detection with domain boundaries
- Reject branches that exit domain
- Use attractors to guide growth toward unperfused regions

### Challenge 2: Mesh quality issues
**Solution**:
- Use quality-guaranteed mesh generator (TetGen with mindihedral flag)
- Refine mesh near vessels
- Smooth mesh using Laplacian smoothing if needed

### Challenge 3: Slow tree generation
**Solution**:
- Reduce number of terminals (50-100 for demo)
- Use simpler cost function
- Skip optimization, use heuristic rules

### Challenge 4: Linear system is singular
**Solution**:
- Check that boundary conditions are properly applied
- Ensure tree is fully connected (no orphan nodes)
- Use direct solver (spsolve) instead of iterative

### Challenge 5: Unrealistic flow values
**Solution**:
- Double-check units (SI units throughout)
- Verify pressure boundary conditions (typical: P_inlet ~ 100 mmHg = 13,300 Pa)
- Check viscosity value (μ = 0.0035 Pa·s)

---

## Conclusion

This implementation guide provides:
1. ✓ Mathematical formulas with physical interpretation
2. ✓ Three algorithm options (CCO, L-system, Hybrid)
3. ✓ Mesh generation strategy using existing tools
4. ✓ Hemodynamic analysis approach
5. ✓ Phase-by-phase roadmap

**Recommended approach for your project**:
- Use **Simplified Hybrid Algorithm** for tree generation (fast, simple, effective)
- Use **TetGen** for meshing (robust, well-documented)
- Implement **network flow analysis** for hemodynamics (linear system, straightforward)

This gives you a complete, implementable project demonstrating:
- Algorithmic geometry generation
- Mesh generation
- Physics-based analysis
- Data-driven validation (compare to literature)

Good luck with your implementation!
