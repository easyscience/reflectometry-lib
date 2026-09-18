# Creating a Model

The main component of an experiment in EasyReflectometry is the `Model`.
This is a description of the `Sample` and the environment in which the
experiment is performed. The `Model` is used to calculate the
reflectivity of the `Sample` at a given set of angles (Q-points). The
resolution functions are used to quantify the experimental uncertainties
in wavelength and angle, allowing the `Model` to accurately describe the
data.

## Model

A `Model` instance contains a `Sample` and variables describing
experimental settings. To be able to compute reflectivities it is also
necessary to have a `Calculator` (interface).

```python
from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import Model
from easyreflectometry.sample import Sample

default_sample = Sample()
model = Model(sample=default_sample, scale=1.0, background=1e-6)

interface = CalculatorFactory()
model.interface = interface
```

This will create a `Model` instance with the `default_sample` and the
environment variables `scale` factor set to 1.0 and a `background` of
1e-6. Following the `interface` is set to the default calculator that is
`Refnx`.

## Resolution Functions

A resolution function enables the EasyReflectometry model to incorporate
the experimental uncertainties in wavelength and incident angle into the
model. In its essence the resolution function controls the smearing to
apply when determining the reflectivity at a given Q-point. For a given
Q-point the smearing to apply is given as a weighted average of the
neighboring Q-point, which weights are by a normal distribution. This
normal distribution is then defined by a Q-point dependent Full Width at
the Half Maximum (FWHM) that is given by the resolution function.

### PercentageFwhm

Often we rely on a resolution function that has a simple functional
dependency of the Q-point. By this is understood that the applied
smearing in a Q-point has a FWHM that is simply a percentage of the
value of the Q-point.

```python
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm

resolution_function = PercentageFwhm(1.1)

m = Model(resolution_function=resolution_function)
```

This will create a `Model` instance where the resolution function is
defined as 1.1% of the Q-point value, which again is the FWHM for the
smearing.

### LinearSpline

Alternatively the FWHM value might be determined and declared directly
for each measured Q-point. When this is the case the provided Q-points
and the corresponding FWHM values can be used to declare a linear spline
function and thereby enable a determination of the reflectivity at an
arbitrary point within the provided range of discrete Q-points.

```python
from easyreflectometry.model import Model
from easyreflectometry.model import LinearSpline

m = Model()

resolution_function = LinearSpline(q_data_points=[0.01, 0.2, 0.31], fwhm_values=[0.001, 0.043, 0.026])

m.resolution_function = resolution_function
```

This will create a `Model` instance where the resolution function
defining the FWHM is determined from a linear interpolation. In the
present case the provided data Q-points are (`[0.01, 0.2, 0.31]`) and
the corresponding FWHM function values are (`[0.001, 0.043, 0.026]`).

## Constraining Parameters

It is often physically motivated to reduce the number of free parameters
in a model by tying parameters together. For example, two layers that
were deposited in the same process step can be expected to have the same
thickness, or the roughness of every interface in a stack can be assumed
to be conformal. Assemblies such as `SurfactantLayer` and `Bilayer`
provide ready-made constraints for their specific chemistry
(`constrain_area_per_molecule`, `conformal_roughness`,
`constrain_multiple_contrast`), but any parameter in a model can be
constrained directly.

### Tying two parameters together

The most common constraint is a simple equality. Here the thickness of
`layer_2` is tied to the thickness of `layer_1`.

```python
from easyreflectometry import constrain_equal

constrain_equal(layer_2.thickness, to=layer_1.thickness)
```

After this call `layer_2.thickness` is no longer an independent
parameter: it immediately takes the value of `layer_1.thickness`,
follows it whenever it changes (including during fitting), and is
removed from the free fit parameters. Only `layer_1.thickness` is varied
by the minimizer.

### Functional constraints

Constraints are not limited to equality. An arbitrary mathematical
expression of one or more parameters can be used, where each placeholder
in the expression is supplied as a keyword argument.

```python
from easyreflectometry import constrain

# layer_2 is always twice as thick as layer_1
constrain(layer_2.thickness, '2 * t', t=layer_1.thickness)

# an SLD that is a fraction-weighted average of two materials
constrain(
    mixed.sld,
    'frac * a + (1 - frac) * b',
    frac=fraction,
    a=solvent.sld,
    b=film.sld,
)
```

Note that constraining a parameter **overwrites its current value, unit
and bounds** with the evaluated expression, and clears its `fixed` flag.
While constrained, the parameter's value and bounds cannot be set
directly.

### Removing a constraint

```python
from easyreflectometry import unconstrain

unconstrain(layer_2.thickness)
```

The parameter keeps its last evaluated value and becomes an independent,
fittable parameter again. Calling `unconstrain` on a parameter that is
not constrained does nothing. The parameter's original bounds and
`fixed` state are **not** restored. They remain whatever the constraint
left behind, so review and reset the bounds before fitting.

### Things to be aware of

- Constraints are directional: the dependent parameter follows the
  independent one, never the other way around. When chaining constraints
  across several objects (as in the multiple-contrast tutorials), make
  sure the chain has a single independent parameter at its root.
- Placeholder names in expressions must be valid Python identifiers and
  not Python keywords. An unmapped name that happens to match a
  mathematical builtin (`e`, `pi`, `sin`, ...) evaluates silently
  instead of raising an error.
- If the model is already attached to a calculator, regenerate the
  bindings after changing constraints so the calculator picks up the new
  dependency graph:

  ```python
  model.generate_bindings()
  ```

- Constraints applied with `constrain`, `constrain_equal` and
  `constrain_to_sum` are preserved when a project is saved and reloaded.
  Dependencies created by calling `make_dependent_on` directly are not,
  and must be re-applied after loading.
