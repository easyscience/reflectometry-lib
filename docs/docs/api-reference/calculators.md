# Calculators

The calculator translates an EasyReflectometry model into a backend
engine (refl1d or refnx) and computes reflectivity.

## Polarized reflectivity

With the refl1d calculator and `include_magnetism` enabled, all four
spin channels are available:

- `polarized_reflectivity_profiles(x_array, model_id)` returns the
  reflectivity of all four channels as a dictionary keyed `'pp'`,
  `'pm'`, `'mp'`, `'mm'` (in that order).
- `polarization_channel` selects which channel `reflectity_profile` —
  and hence fitting — uses (default `'pp'`).
- `magnetic_sld_profile(model_id)` returns the nuclear and magnetic
  scattering length density profiles as a tuple `z`, `sld(z)`, `rhoM(z)`
  (magnetic SLD) and `thetaM(z)` (magnetic angle).

Note that `polarization_channel` is state of the currently active
calculator instance, not of a model or dataset: it affects every
subsequent calculation using that calculator, and switching calculators
via the factory constructs a fresh instance, which resets the channel
(along with `include_magnetism`). Calculators without magnetism support
(refnx) raise `NotImplementedError` when magnetism is enabled.

::: easyreflectometry.calculators.polarization

::: easyreflectometry.calculators.calculator_base
