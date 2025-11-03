# Project 2: Physics-Based Liver Vascular Modeling

## Overview
This project implements physics-based vascular tree generation and hemodynamic analysis for liver vasculature, completing the **Programming Option (Option 2)** for CMU's Image-Based Computational Modeling course.

**Topic**: Physics-based and data-driven modeling of vascular modeling in liver, limb or head (focused on liver)

## Project Structure

### Documentation
1. **[literature_review.md](literature_review.md)** - Comprehensive review of 6 technical papers
   - Analysis of tree generation algorithms (CCO, optimization-based, L-systems)
   - Murray's law and hemodynamic principles
   - Comparative analysis and recommendations
   - Key formulas and validation approaches

2. **[implementation_guide.md](implementation_guide.md)** - Mathematical formulas and implementation roadmap
   - Detailed mathematical derivations (Murray's law, Poiseuille flow, etc.)
   - Three algorithm options with pseudocode
   - Mesh generation strategies
   - Hemodynamic analysis methods
   - Phase-by-phase implementation plan

### Python Scripts

3. **[vascular_tree_generator.py](vascular_tree_generator.py)** - Vascular tree generation
   - Recursive bifurcation algorithm with Murray's law
   - Space-filling tree growth in 3D liver domain
   - Generates trees with realistic branching patterns
   - Visualization and export capabilities

4. **[liver_mesh_hemodynamics.py](liver_mesh_hemodynamics.py)** - Hemodynamic analysis
   - Converts tree to network graph
   - Solves pressure-flow distribution using Poiseuille's law
   - Computes shear stress and Reynolds numbers
   - Visualization of hemodynamic quantities
   - VTK export for ParaView

## Quick Start

### Prerequisites
```bash
pip install numpy matplotlib scipy
pip install meshio  # optional, for VTK export
```

### Running the Code

**Step 1: Generate vascular tree**
```bash
python vascular_tree_generator.py
```
This generates:
- `vascular_tree.json` - Tree data structure
- `tree_generation.png` - Tree colored by generation
- `tree_radius.png` - Tree colored by radius
- `tree_resistance.png` - Tree colored by resistance

**Step 2: Run hemodynamic analysis**
```bash
python liver_mesh_hemodynamics.py
```
This generates:
- `hemodynamics_pressure.png` - Pressure distribution
- `hemodynamics_flow.png` - Flow rate distribution
- `hemodynamics_shear.png` - Shear stress distribution
- `liver_network.vtk` - VTK file for ParaView (if meshio installed)

## Key Algorithms Implemented

### 1. Vascular Tree Generation
- **Method**: Simplified hybrid recursive bifurcation
- **Features**:
  - Murray's law for vessel radii: r₀³ = r₁³ + r₂³
  - Physiologically realistic branching angles (25-45°)
  - Space-filling growth within liver domain
  - Adjustable asymmetry ratios

### 2. Hemodynamic Analysis
- **Method**: Network flow solver using Poiseuille's law
- **Equations**:
  - Flow: Q = (π r⁴ ΔP) / (8 μ L)
  - Resistance: R = 8 μ L / (π r⁴)
  - Conservation: Σ Q_in = Σ Q_out at each node
- **Outputs**: Pressure, flow rate, shear stress, Reynolds number

## Results & Validation

### Expected Tree Statistics
- **Segments**: ~100-500 (depending on max_generation parameter)
- **Generations**: 6-8 levels
- **Radius range**: 0.1-5.0 mm (root to terminals)
- **Total volume**: ~500-2000 mm³

### Expected Hemodynamics
- **Pressure drop**: ~90 mmHg (100 → 10 mmHg)
- **Total flow**: ~50-200 ml/min (typical hepatic artery flow)
- **Shear stress**: 1-10 Pa (physiological range)
- **Reynolds number**: <2300 (laminar flow)

## Comparison to Literature

The implementation aligns with:
1. **Correa-Alfonso et al. (2022)**: Algorithmic tree growth + meshing approach
2. **Jessen et al. (2022)**: Murray's law application for radius optimization
3. **Taebi et al. (2020)**: Hemodynamic analysis using Poiseuille flow

## Customization

### Modify Tree Parameters
In `vascular_tree_generator.py`, adjust:
```python
tree = generate_simple_tree(
    root_radius=5.0,        # Root vessel radius (mm)
    max_generation=7,       # Branching depth
    domain_size=100.0       # Domain size (mm)
)
```

### Modify Boundary Conditions
In `liver_mesh_hemodynamics.py`, adjust:
```python
network = solve_hemodynamics(
    inlet_pressure=13332.0,   # Pa (100 mmHg)
    outlet_pressure=1333.0,   # Pa (10 mmHg)
    viscosity=0.0035          # Pa*s
)
```

## Report Outline (8-12 pages)

Suggested structure based on project requirements:

### 1. Introduction (1-2 pages)
- Motivation: Importance of liver vascular modeling
- Applications: Surgery planning, radiotherapy, perfusion analysis
- Project goal: Implement algorithmic tree generation + hemodynamic analysis

### 2. Background & Literature Review (2-3 pages)
- Summary of 6 papers from `literature_review.md`
- Comparison of approaches (CCO, optimization, L-systems)
- Justification for chosen method

### 3. Methods (3-4 pages)
- **Tree generation algorithm**: Recursive bifurcation with Murray's law
  - Pseudocode
  - Mathematical formulation
- **Hemodynamic solver**: Network flow with Poiseuille's law
  - Linear system setup
  - Boundary conditions
- **Implementation details**: Python, libraries, data structures

### 4. Results (2-3 pages)
- **Tree statistics**: Number of segments, generations, radii distribution
- **Hemodynamic results**: Pressure maps, flow distribution, shear stress
- **Validation**: Comparison to literature values
- **Computational performance**: Runtime, scaling analysis

### 5. Discussion (1-2 pages)
- **Strengths**: Fast, implementable, physiologically plausible
- **Limitations**: Simplified geometry, no patient-specific data, straight cylinders
- **Future work**: Image-based trees, fluid-structure interaction, pathological models

### 6. Conclusion (0.5-1 page)
- Summary of achievements
- Broader impact

## Presentation Outline (10 minutes + 2 min Q&A)

**Slide suggestions**:
1. Title & motivation
2. Problem statement
3. Literature review (brief)
4. Algorithm overview (tree generation)
5. Algorithm overview (hemodynamics)
6. Implementation highlights
7. Results: Tree visualization
8. Results: Hemodynamic analysis
9. Validation & comparison to literature
10. Limitations & future work
11. Conclusions
12. Questions

## Advanced Extensions

If you want to go further:
1. **Optimize tree using cost function** (implement Jessen et al. approach)
2. **Add pulsatile flow** (time-dependent boundary conditions)
3. **Include vessel compliance** (elastic walls)
4. **Generate tetrahedral mesh** (using TetGen or pygalmesh)
5. **Add tumor model** (obstruction, altered perfusion)

## Troubleshooting

**Tree grows outside domain**:
- Reduce `max_generation` parameter
- Decrease branching angle
- Adjust `domain_size`

**Unrealistic flow values**:
- Check pressure units (Pa, not mmHg)
- Verify viscosity (0.0035 Pa*s)
- Ensure boundary conditions are correct

**Linear system is singular**:
- Verify tree is fully connected
- Check that inlet/outlet nodes are properly identified
- Use direct solver (spsolve)

## References

See [literature_review.md](literature_review.md) for full citations of 6 technical papers.

## Contact

For questions about this project implementation, refer to:
- Implementation guide: `implementation_guide.md`
- Course materials: CMU 24658/42640
- Technical papers: `technical_papers.md`

---

**Good luck with your project presentation and report!**
