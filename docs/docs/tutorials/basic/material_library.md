# Defining Materials

In order to support a wide range of applications (and to build complex
assemblies) there are a few different types of material that can be
utilised in EasyReflectometry. These can include constraints or enable
the user to define the material based on chemical or physical
properties. Full API documentation for the
`easyreflectometry.sample.elements.material` module is also available,
but here we will give some simple uses for them.

## Material

The simplest type of material that is available is the `Material`. This
allows the user to define a single type of material, with a real and
imaginary component to the scattering length density. The construction
of a `Material` is achieved as shown below.

```python
from easyreflectometry.sample import Material

boron = Material(sld=6.908, isld=-0.278, name='Boron')
```

The above object will have the properties of `sld` and `isld`, which
will have values of `6.908 1/angstrom^2` and `-0.278 1/angstrom^2`
respectively. As is shown in the tutorials, a material can be used to
construct a `Layer` from which
[slab models](https://www.reflectometry.org/isis_school/3_reflectometry_slab_models/the_slab_model.html)
are created.

## MaterialDensity

In addition to defining a material by its scattering length density, it
may be useful to define a material by the mass density and chemical
formula. This is possible with the `MaterialDensity` material type,
which uses the scattering length and atomic mass from the chemical
formula and the density to determine the scattering length density. It
is then possible to vary the density, which defines the scattering
length density in turn. The `MaterialDensity` material can be created as
follows.

```python
from easyreflectometry.sample import MaterialDensity

chemical_structure = 'SiO2'
si = MaterialDensity(chemical_structure=chemical_structure, density=2.65, name='SiO2 Material')
```

The density should be in units of grams per cubic centimeter and the
scattering length is calculated from `'SiO2'`.

By default the `sld` and `isld` of a `MaterialDensity` are _dependent_
parameters, recomputed from the density, the formula's scattering length
and its molecular weight whenever any of those change - so `density` is
the parameter to vary in a fit, and assigning to `sld` directly is not
possible. This coupling can be switched off per material with the
`sld_coupled` property:

```python
si.sld_coupled = False  # sld/isld become independent, keep their values
si.sld.fixed = False  # ...and can now be fitted directly
```

While decoupled, changes to `density` (or the formula) no longer
propagate to the SLD, and the density, molecular weight and scattering
length no longer affect the reflectivity. Setting `sld_coupled = True`
restores the dependency and **recalculates** `sld`/`isld` from the
current formula and density, discarding any manually set or fitted
values. The coupling state - and, when decoupled, the manual SLD
values - survive serialization (`as_dict`/`from_dict`); dictionaries
from before this feature restore as coupled.

Assigning a new `chemical_structure` updates both the scattering length
and the molecular weight; a formula that does not parse to at least one
known atom raises `ValueError` and leaves the material unchanged.

Note that `molecular_weight` is a read-only descriptor, not a fit
parameter: it is fully determined by the formula.

## MaterialSolvated

Sometimes it is desirable to have a layer that consists of a material
and a solvent in some ratio. An example of this is shown in the
solvation tutorial, where a polymer film solvated with D2O is modelled.
To produce a material that is described by such a mixture, there is
`MaterialSolvated`. This is constructed from two constituent `Materials`
and the fractional amount of the material in the solvent. So to produce
a `MaterialSolvated` that is 20% D2O in a polymer, the following is
used.

```python
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialSolvated

polymer = Material(sld=2.0, isld=0.0, name='Polymer')
d2o = Material(sld=6.36, isld=0, name='D2O')

solvated_polymer = MaterialSolvated(material=polymer, solvent=d2o, solvent_fraction=0.2, name='Solvated Polymer')
```

For the `solvated_polymer` object, the `sld` will be
`2.872 1/angstrom^2` (the weighted average of the two scattering length
densities). The `MaterialSolvated` includes a constraint such that if
the value of either constituent scattering length densities (both real
and imaginary components) or the fraction changes, then the resulting
material `sld` and `isld` will change appropriately.
