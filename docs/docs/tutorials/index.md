---
icon: material/school
---

# :material-school: Tutorials

This section presents a collection of **Jupyter Notebook** tutorials
that demonstrate how to use EasyReflectometry for various tasks. These
tutorials serve as self-contained, step-by-step **guides** to help users
grasp the workflow of data analysis using EasyReflectometry.

Instructions on how to run the tutorials are provided in the
[:material-cog-box: Installation & Setup](../installation-and-setup/index.md#how-to-run-tutorials)
section of the documentation.

## Getting Started

- [Creating a Model](basic/model.md) – Learn how to define a
  reflectometry model with sample, scale, background, and resolution
  functions.
- [Defining Materials](basic/material_library.md) – Explore different
  material types: `Material`, `MaterialDensity`, `MaterialSolvated`, and
  `MaterialMixture`.
- [Defining Layers](basic/layer_library.md) – Understand layer types
  including `Layer` and `LayerAreaPerMolecule`.
- [Creating Assemblies](basic/assemblies_library.md) – Build complex
  structures with `Multilayer`, `RepeatingMultilayer`, and
  `SurfactantLayer`.

## Simulation

These are basic simulation examples using the EasyReflectometry library.

- [Bilayer Simulation](simulation/bilayer.ipynb)
- [Magnetism Simulation](simulation/magnetism.ipynb)
- [Resolution Functions](simulation/resolution_functions.ipynb)

## Fitting

These are basic fitting examples using the EasyReflectometry library.

- [Simple Fitting](fitting/simple_fitting.ipynb)
- [Repeating Multilayer Fitting](fitting/repeating.ipynb)
- [Monolayer Fitting](fitting/monolayer.ipynb)
- [Solvated Material Fitting](fitting/material_solvated.ipynb)

## Advanced Fitting

These are advanced fitting examples using the EasyReflectometry library.

- [Multi-Contrast Fitting](advancedfitting/multi_contrast.ipynb)
- [Polarized Fitting](advancedfitting/polarized_fitting.ipynb) –
  Magnetic samples, all four spin channels, the spin-resolved depth
  profile and simultaneous fitting of several channels against one
  model.
- [Constraints & Inequalities](advancedfitting/constraints.ipynb) –
  Equality constraints, derived read-only parameters (`total_thickness`,
  `constrain_to_sum`) and inequality constraints enforced during BUMPS
  fits.
- [Bayesian Fitting](advancedfitting/bayesian_bumps.ipynb) – DREAM MCMC
  sampling through `MultiFitter.mcmc_sample()`, convergence diagnostics,
  posterior summaries and posterior-predictive bands.
