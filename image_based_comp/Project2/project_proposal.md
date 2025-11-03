# Project 2 Proposal: Physics-Based Liver Vascular Modeling

**Course**: 24658/42640 Image-Based Computational Modeling and Analysis
**Team Member(s)**: [Your Name]
**Option**: Programming (Option 2)
**Topic**: Physics-based and data-driven modeling of vascular modeling in liver

## Overview

This project will implement algorithmic vascular tree generation and hemodynamic analysis for liver vasculature. The liver's complex vascular network, comprising hepatic arterial, portal venous, and hepatic venous trees, plays a critical role in organ function, surgical planning, and disease treatment. Accurate computational models of liver vasculature enable improved radiation dosimetry, perfusion analysis, and therapy planning.

## Objectives

The primary goal is to develop Python-based tools that:
1. Generate synthetic vascular trees using physics-based growth algorithms
2. Create tetrahedral meshes with embedded vessels suitable for finite element analysis
3. Perform hemodynamic simulations to compute pressure and flow distributions
4. Validate results against physiological data from literature

## Methodology

The implementation will follow the constrained constructive optimization (CCO) approach described in Correa-Alfonso et al. (2022), combined with Murray's law for optimal vessel radii. The tree generation algorithm will recursively create bifurcations while satisfying the principle r₀³ = r₁³ + r₂³, which minimizes metabolic and pumping costs. Hemodynamic analysis will employ Poiseuille's law for 1D flow in vessels, solving a linear system for pressure distribution across the vascular network.

## Expected Outcomes

The project will deliver:
- Two Python scripts: one for vascular tree generation, one for mesh generation and hemodynamic analysis
- Visualizations showing tree morphology, pressure maps, flow distributions, and shear stress
- An 8-12 page report comparing results to literature values (typical hepatic artery flow: 100-200 ml/min, pressure drop: 90 mmHg)
- Validation against morphological metrics from human liver corrosion cast data

## Technical Papers

Six relevant papers have been identified covering algorithmic tree generation (Correa-Alfonso 2022, Jessen 2022), hemodynamic simulation (Taebi 2020), and multiscale coupling (Ebrahem 2023, 2025; Rohan 2016). These provide theoretical foundations and validation benchmarks for the implementation.

---

**Word Count**: ~295 words
