# Recommended Papers

## 1. [A mesh-based model of liver vasculature: Implications for improved radiation dosimetry to liver parenchyma for radiopharmaceuticals](https://pubmed.ncbi.nlm.nih.gov/35416550/) (Correa-Alfonso et al., 2022)

* Developed computer-generated models of hepatic arterial (HA), portal venous (HPV) and venous (HV) trees inside the reference adult liver lobe volumes.
* They algorithmically grow vascular trees (straight cylinders) inside lobes, update hemodynamic/geometry parameters on each iteration.
* Then tetrahedralize the volume including these vessels for coupling to radiation transport code.
* **Why useful:** Directly relevant to your mesh/vasculature topic in liver. It includes tree generation + meshing.
* **Potential algorithm to implement:** Their tree-growth algorithm + tree meshing could be a good choice for Option 2.

## 2. [Rigorous mathematical optimization of synthetic hepatic vascular trees](https://arxiv.org/abs/2202.04406) (Jessen, Steinbach, Debbaut, Schillinger, 2022)

* Introduces a framework for generating synthetic vascular trees via a nonlinear optimization formulation (not just heuristics).
* Validated against human liver corrosion cast data to match structural metrics.
* **Why useful:** High rigor for algorithmic tree generation; good if you like optimization/algorithm focus.

## 3. [Computational Modeling of the Liver Arterial Blood Flow for Microsphere Therapy: Effect of Boundary Conditions](https://www.mdpi.com/2306-5354/7/3/64) (Taebi et al., 2020)

* CFD simulation on hepatic arterial tree extracted from imaging, studies how outlet boundary conditions affect flow and microsphere distribution.
* **Why useful:** If you want to focus on physics-based hemodynamics (not just geometry) of liver vasculature.

## 4. [Connecting continuum poroelasticity with discrete synthetic vascular trees for modeling liver tissue](https://arxiv.org/pdf/2306.07412) (Ebrahem, Jessen, Mika, Schillinger, 2023)

* Coupling discrete vascular tree models with a continuum poroelastic model of liver tissue.
* Could be more complex, but if you are ambitious, this merges structural deformation + perfusion.
* **Why useful:** Integrates multiple physics (perfusion + mechanics) which may align well with your project theme.

## 5. [Modeling of the contrast-enhanced perfusion test in liver based on the multi-compartment flow in porous media](https://arxiv.org/abs/1605.09162) (Rohan, Lukeš, Jonášová, 2016)

* Multi-scale model of liver perfusion combining 1D flows in major vessels and 3D porous media flow in parenchyma.
* **Why useful:** If you want to focus on perfusion (data-driven/physics) rather than tree generation, good alternative.

## 6. [A multiscale-multiphysics framework for modeling organ-scale liver regrowth](https://www.sciencedirect.com/science/article/pii/S0022509625000894) (Ebrahem et al., 2025)

* A newer paper (2025) integrating synthetic vascular tree generation + homogenized perfusion + poroelastic growth, applied to liver post-resection.
* **Why useful:** Shows state-of-the-art multiphysics; may be more advanced than required, but good for survey/background.