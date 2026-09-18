# Creating Multilayers and Surfactant Layers

EasyReflectometry is designed to be used with a broad range of different
assemblies. Assemblies are collective layers behaving as a single
object, for example, a multilayer or a surfactant layer. These
assemblies offer flexibility for the user and enable more powerful
analysis by making chemical and physical constraints available with
limited code. In this page, we will document the assemblies that are
available with simple examples of the constructors that exist. Full API
documentation is also available for the
`easyreflectometry.sample.assemblies` module. For custom constraints
between arbitrary parameters, see _Constraining Parameters_ in the
[Model tutorial](model.md).

## Multilayer

This assembly should be used for a series of layers that should be
thought of as a single object. For example, in the simple fitting
tutorial this assembly type is used to combine the silicon and silicon
dioxide layer that is formed into a single object. All of the separate
layers in these objects will be fitted individually, i.e. there are no
constraints present, however, there is some cognitive benefit to
grouping layers together.

To create a `Multilayer` object, we use the following construction.

```python
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer

si = Material(sld=2.07, isld=0, name='Si')
sio2 = Material(sld=3.47, isld=0, name='SiO2')
si_layer = Layer(material=si, thickness=0, roughness=0, name='Si layer')
sio2_layer = Layer(material=sio2, thickness=30, roughness=3, name='SiO2 layer')

subphase = Multilayer(layers=[si_layer, sio2_layer], name='Si/SiO2 subphase')
```

This will create a `Multilayer` object named `subphase` which we can use
in some `Structure` for our analysis.

### Conformal thickness and roughness

Layers grown in a single process step often share a thickness or a
roughness. A `Multilayer` can tie every one of its layers to the front
layer, which removes the followers from the fit and leaves a single free
parameter per tied quantity.

```python
subphase = Multilayer(
    layers=[si_layer, sio2_layer],
    name='Si/SiO2 subphase',
    conformal_roughness=True,
    conformal_thickness=True,
)
```

Both toggles are also plain properties, so they can be switched on and
off after construction.

```python
subphase.conformal_roughness = True
subphase.conformal_roughness = False  # releases the followers again
```

The ties are ordinary parameter dependencies, and the assembly stores
the state of both toggles when it is serialized, so they are rebuilt
when a project is loaded. Releasing a tie leaves the follower at its
last value; as with any constraint, its bounds are whatever the
constraint left behind, so review them before fitting. Constraints
between arbitrary parameters — including a fixed total thickness — are
covered in _Constraining Parameters_ in the [Model tutorial](model.md)
and in the [Constraints tutorial](../advancedfitting/constraints.ipynb).

## RepeatingMultilayer

The `RepeatingMultilayer` assembly type is an extension of the
`Multilayer` for the analysis of systems with a multilayer that has some
number of repeats. This assembly type imposes some constraints,
specifically that all of the repeats have the exact same structure (i.e.
thicknesses, roughnesses, and scattering length densities), which brings
with it some computational saving as the reflectometry coefficients only
need to be calculated once for this structure and propagated for the
correct number of repeats. There is a tutorial that discusses the
utilisation of this assembly type for a nickel-titanium multilayer
system.

The creation of a `RepeatingMultilayer` object is very similar to that
for the `Multilayer`, with the addition of a number of repetitions.

```python
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import RepeatingMultilayer

ti = Material(sld=-1.9493, isld=0, name='Ti')
ni = Material(sld=9.4245, isld=0, name='Ni')
ti_layer = Layer(material=ti, thickness=40, roughness=0, name='Ti Layer')
ni_layer = Layer(material=ni, thickness=70, roughness=0, name='Ni Layer')
ni_ti = RepeatingMultilayer(layers=[ti_layer, ni_layer], repetitions=10, name='Ni/Ti Multilayer')
```

The number of repeats is a parameter that can be varied in the
optimisation process, however given this is a value that depends on the
synthesis of the sample this is unlikely to be necessary.

A `RepeatingMultilayer` is a `Multilayer`, so it accepts the
`conformal_thickness` and `conformal_roughness` arguments described
above, and persists them in the same way.

```python
ni_ti = RepeatingMultilayer(
    layers=[ti_layer, ni_layer],
    repetitions=10,
    name='Ni/Ti Multilayer',
    conformal_roughness=True,
)
```
