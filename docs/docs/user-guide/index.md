---
icon: material/book-open-variant
---

# :material-book-open-variant: User Guide

This section provides an overview of the **core concepts**, **key
parameters** and **workflow steps** required for using EasyReflectometry
effectively.

## Glossary

The following serves to clarify what we mean by the terms we use in this
project.

### Sample

A sample is an ideal representation of the full physical setup. This
includes the layer(s) under investigation, the surrounding superphase,
and the subphase.

### Calculator

A calculator is the physics engine which calculates the reflectivity
curve from our inputted sample parameters. We rely on third party
software to provide the necessary calculators. Different calculators
might have different capabilities and limitations.

EasyReflectometry offers two calculation engines:

- [**refnx**](https://refnx.readthedocs.io/)
- [**Refl1D**](https://refl1d.readthedocs.io/en/latest/)

Refl1D is the engine to choose for magnetic samples and polarized data;
refnx does not model magnetism.

We are working to add more, in particular
[**GenX**](https://aglavic.github.io/genx/doc/).

### Model

A model combines a sample and calculator. The model is also responsible
for including instrumental effects such as background, scale, and
resolution.

### Assemblies

Assemblies are collections of layers that are used to represent a
specific physical setup. Examples include:

- **Multilayer** – A series of layers grouped as a single object
- **RepeatingMultilayer** – A multilayer with a fixed number of repeats
- **SurfactantLayer** – A layer defined by area per molecule and
  chemical formula
- **GradientLayer** – A layer with a graded scattering length density
  profile

### Elements

Elements are the building blocks that are required to construct a
sample.

**Layers** are basic elements used to represent a single layer of
material with a thickness and a roughness:

- `Layer` – Standard layer with material, thickness, and roughness
- `LayerAreaPerMolecule` – Layer defined by area per molecule and
  chemical formula

**Materials** are the most basic elements and are used to represent a
material with given physical properties:

- `Material` – Simple material defined by SLD (real and imaginary)
- `MaterialDensity` – Material defined by mass density and chemical
  formula
- `MaterialSolvated` – Material mixed with a solvent in a given ratio
- `MaterialMixture` – Mixture of two materials

### Fitting

Fitting helpers and objective functions. The `MultiFitter` supports
several objective modes for handling reflectometry data during fitting,
especially when measured variances are non-positive.

## Getting Started

To use EasyReflectometry in a project:

```python
import easyreflectometry
from easyreflectometry.sample import Material, Layer
from easyreflectometry.model import Model
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.plot import plot

# Define your Material
material = Material(...)

# Create a Layer
layer = Layer(material=material, ...)

# Make a Sample out of the Layer
sample = Sample(layer, ...)

# Define a Model of the experiment
model = Model(
    sample=sample,
    scale=1,
    background=1e-6,
    ...
)

# Set parameter bounds for fit
...

# Perform the fit and plot
fitter = MultiFitter(model)
analysed = fitter.fit(data)

plot(analysed)
```

Details of specific usage of EasyReflectometry can be found in the
[Tutorials](../tutorials/index.md).

## Volume Fraction (Occupancy) Profiles

The SLD profile of a sample is a single blended curve. A **volume
fraction profile** separates it into its components: how much of the
volume at each depth is lipid head group, tail, solvent, substrate and
so on. It is the kind of plot shown in the
[refnx occupancy documentation](https://refnx.readthedocs.io/en/latest/occupancy.html)
and used, for example, to follow an antibiotic into a membrane.

```python
from easyreflectometry.plot import plot_volume_fraction

profile = sample.volume_fraction_profile()  # no calculator needed
print(profile.labels)                       # component key -> label
plot_volume_fraction(profile)               # one filled trace per component

# Presentation grouping: merge and rename traces
grouped = profile.grouped({'Heads': ['C10H18NO8P'], 'Water': ['D2O']})
plot_volume_fraction(grouped, style='line')
```

`Project.volume_fraction_data_for_model_at_index` returns the same
profiles as `DataSet1D` objects for use in a GUI or a report.

How it is built:

- Every layer is decomposed into its materials: a `MaterialSolvated`
  or `MaterialMixture` contributes its two parts weighted by its
  fraction, a `LayerAreaPerMolecule` contributes its molecule and its
  solvent, a `RepeatingMultilayer` is expanded, and a `GradientLayer`
  is read as a linear *mixing* gradient between its two end materials
  (an SLD ramp alone does not identify a mixture; this reading is a
  definition, and a gradient layer edited after construction is
  refused).
- The per-layer fractions are smeared across each interface with the
  same error-function kernel as the plotted SLD profile. The fractions
  therefore sum to one at every depth, and
  `sum(rho_c * phi_c(z))` reproduces the SLD profile whenever each
  component has one SLD throughout.
- `z = 0` is the interface between the superphase and the first film
  layer, increasing into the sample; this is the refnx calculator's SLD
  convention and the default grid is the same 500 points.

Grouping. Materials the calculator cannot tell apart (same name, same
SLD) are merged into one component by default
(`merge_equivalent=True`), so the several default solvent objects of a
`Bilayer` appear as one "D2O" trace and its two head groups as one
trace. Materials that share a name but differ in SLD are never merged
automatically; use `profile.grouped(...)` to merge anything else, such
as D₂O and H₂O into "Water" across contrasts.

Caveats worth knowing:

- The error-function kernel is the *plotting* convention shared with
  the SLD profile, not the Névot–Croce factor the reflectivity
  calculation uses. The curves are a model-derived in-plane average,
  not a measured concentration profile.
- Fractions sum to one but need not stay within `[0, 1]`: with very
  different roughness on the two sides of a thin layer a fraction dips
  below zero. The values are reported unchanged, a
  `VolumeFractionWarning` says where, and the plot shows the dip. The
  remedy is in the roughness parameters, not in the plot.
- A `solvent_fraction` may describe lateral patchiness rather than
  solvation; the in-plane average is drawn either way.
- For a `LayerAreaPerMolecule` the fractions are those of the current
  material model, in which the solvent fraction acts as coverage of a
  layer whose molecule SLD is `b / (thickness × area per molecule)`.
- The default depth range only looks at the two outer roughnesses; pass
  `z=` (or `max_delta_z=`) for a wider or finer grid.

## Objective Functions and Non-Positive Variance Handling

`MultiFitter` supports several objective modes for handling
reflectometry data during fitting, especially when measured variances
are non-positive.

The default objective is `hybrid`. This uses ordinary weighted least
squares for points with positive variance and applies a Mighell-style
substitution only to points whose variance is non-positive. The older
`legacy_mask` mode drops non-positive-variance points before fitting.
The `mighell` mode applies the Mighell transform to every point.

### Mighell Objective

The full `mighell` objective follows the algebraic form of the
$\chi^2_\gamma$ statistic described by Mighell for Poisson-distributed
count data:

$$
\chi^2_\gamma =
\sum_i \frac{[n_i + \min(n_i, 1) - m_i]^2}{n_i + 1}
$$

where $n_i$ are observed counts and $m_i$ are model values.

In EasyReflectometry this is implemented as a weighted least-squares
problem. For each observed value $y_i$ the fitted target is shifted to

$$
y_{\mathrm{eff},i} = y_i + \min(y_i, 1)
$$

and the effective uncertainty is

$$
\sigma_i = \sqrt{y_i + 1}
$$

so the minimized objective is

$$
\sum_i \left(\frac{y_{\mathrm{eff},i} - f_i}{\sigma_i}\right)^2 =
\sum_i \frac{[y_i + \min(y_i, 1) - f_i]^2}{y_i + 1}
$$

### Scope and Interpretation

Mighell's statistic was derived for Poisson-distributed count data. In
reflectometry workflows, the fitted values are usually normalized
reflectivities or intensities rather than raw counts. They may already
have been processed, scaled, background-corrected, or otherwise
transformed before they reach the fitter.

This distinction matters when interpreting the result. The full
`mighell` objective is not only a reweighting of residuals; it also
changes the fitted target from $y$ to $y + \min(y, 1)$. For values
between zero and one, this can substantially increase the target value.
A fit can therefore have a good Mighell objective value while looking
poorer against the originally plotted reflectivity curve, or while
having a worse classical chi-square.

For reflectometry data, `hybrid` is generally the recommended
compromise: it preserves ordinary weighted least-squares behavior where
positive variances are available, while still allowing
non-positive-variance points to contribute through the Mighell-style
substitution.

### Objective Modes

- **`hybrid`** (default): Use standard weighted least squares for points
  with positive variance and apply the Mighell substitution only where
  variance is non-positive.
- **`mighell`**: Apply the Mighell transform to all points. The reported
  objective chi-square is evaluated in transformed objective space and
  should not be interpreted as a classical chi-square against the
  original reflectivity values.
- **`legacy_mask`**: Remove non-positive-variance points before fitting
  and use standard weighted least squares for the remaining points.
- **`auto`**: Alias for `hybrid`.

### Fit Metrics

The fitter exposes both objective-space and classical fit metrics after
fitting. `objective_chi2` and `objective_reduced_chi` describe the
minimized objective value, while `classical_chi2` and
`classical_reduced_chi` describe the fit quality against the original
reflectivity values (with non-positive-variance points excluded).
