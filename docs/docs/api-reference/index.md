---
icon: material/code-braces-box
---

# :material-code-braces-box: API Reference

This section contains the auto-generated reference detailing the
functions and modules available in EasyReflectometry.

## Model

Model is a sample, a background and a resolution.

- [Model](model.md)

## Sample

Sample is built from assemblies.

- [Sample](sample.md)

## Project

Project provides a higher-level interface for managing models,
experiments, and ORSO import.

- [Project](project.md)

## Fitting

Fitting helpers and objective functions.

- [Fitting](fitting.md)

## Constraints

Equality constraints and derived parameters tie parameters together and
remove the dependent one from the fit. Inequality constraints keep every
parameter free and are enforced as penalties on the BUMPS fit problem.

- [Constraints](constraints.md)
- [Inequality Constraints](inequality_constraints.md)

## Bayesian Analysis

Posterior analysis of a DREAM sampling run: convergence diagnostics,
credible intervals and correlation plots.

- [Bayesian Analysis](bayesian.md)

## Calculators

The backend engines (refl1d, refnx) that turn a model into a
reflectivity curve, including the polarized interface.

- [Calculators](calculators.md)

## Parameter Limits

Default `min` / `max` windows applied to parameters that have none.

- [Parameter Limits](limits.md)

## Assemblies

Assemblies are collections of layers that are used to represent a
specific physical setup.

- [Multilayer](assemblies/multilayer.md)
- [Repeating Multilayer](assemblies/repeating_multilayer.md)
- [Surfactant Layer](assemblies/surfactant_layer.md)
- [Gradient Layer](assemblies/gradient_layer.md)

## Elements

Elements are the building blocks that are required to construct a
sample.

### Layers

Layers are basic elements and used to represent a single layer of
material with a thickness and a roughness.

- [Layer](elements/layer.md)
- [Layer Area Per Molecule](elements/layer_area_per_molecule.md)
- [Layer Magnetism](elements/layer_magnetism.md)

### Materials

Materials are the most basic elements and are used to represent a
material with given physical properties.

- [Material](elements/material.md)
- [Material Density](elements/material_density.md)
- [Material Mixture](elements/material_mixture.md)
- [Material Solvated](elements/material_solvated.md)

## Data

Collection of helper functions.

- [Data](data.md)
- [ORSO](orso.md)

## Reporting

Summaries of a project and its fit results, and the plotting helpers.

- [Summary](summary.md)
- [Plotting](plot.md)
