# Unreleased

## Project persistence

- `Project.save_as_json` now writes atomically: the file is serialized
  first, written to a temporary file in the project directory, synced
  and moved into place, so a failure while serializing or writing leaves
  a previously saved `project.json` untouched. An existing file keeps
  its permissions; a new one gets the ordinary umask-derived mode
  instead of the owner-only mode of a temporary file.
- Behaviour change: `save_as_json`, `create` and `load_from_json` raise
  instead of printing to stdout. `save_as_json` raises `FileExistsError`
  when the file exists and `overwrite` is False, `ValueError` when the
  project cannot be serialized (a `TypeError` from the encoder is
  reported as `ValueError` too) and `OSError` when the file cannot be
  written. `create` raises `FileExistsError` when the project directory
  already exists and `load_from_json` raises `FileNotFoundError` when
  there is no file at the path. Scripts that relied on these calls
  silently continuing must catch the exceptions.

## Models

- `ModelCollection(interface=...)` with no models now builds its default
  model correctly. The default was passing the calculator interface
  positionally into `Model`, where it landed as the `sample` argument,
  so constructing a collection with a real calculator and no models
  failed.

## Parameter constraints

- New equality-constraint helpers `constrain`, `constrain_equal`,
  `unconstrain`, `constrain_to_sum` and `derived_parameter`
  (`easyreflectometry.constraints`), thin wrappers over the EasyScience
  parameter-dependency mechanism. Constraints created through these
  helpers survive project save/load; raw `make_dependent_on` calls do
  not. A standalone `derived_parameter` is session-only: it has no
  structural path and a saved project cannot reference it.
- New inequality constraints
  (`easyreflectometry.inequality_constraints`): declarative
  `InequalitySpec` objects (`t_head < t_tail`, `t1 + t2 <= 90`)
  registered on the project (`Project.add_inequality_constraint` and
  friends) and enforced as penalties on the BUMPS fit problem, including
  via `MultiFitter.for_experiments`-built fitters driven through the raw
  `easy_science_multi_fitter.fit(...)`. Engines that cannot enforce them
  (LMFit, DFO-LS) raise instead of silently dropping physics. Older
  project files without the new keys load unchanged; files saved with
  constraints keep the file format at 2 (old readers ignore the additive
  keys and lose the constraints).
- New `clamp_sum_partners` / `restore_sum_partners`: a
  `constrain_to_sum` remainder can be driven negative by a fit that
  pushes the partners past the total (a layer of negative thickness).
  `clamp_sum_partners` caps each partner's `max` at the headroom it
  leaves, sharing the slack in proportion to the current values;
  `restore_sum_partners` hands the original maxima back and is
  idempotent. The stashed maxima are persisted by structural path, so
  the round trip survives project save/load. New `is_constrained_to_sum`
  reports whether a parameter carries a `constrain_to_sum` dependency,
  including after a reload.
- New `easyreflectometry.UnitError`, raised by `check_units` for a unit
  problem in an inequality constraint. It subclasses `ValueError`, so
  existing `except ValueError` handlers keep working, but callers no
  longer have to match message substrings.
- New `Model.total_thickness`: a read-only derived parameter equal to
  the summed thickness of the layers between superphase and subphase,
  rebuilt whenever the layer structure changes. New
  `conformal_thickness` / `conformal_roughness` toggles on assemblies,
  also accepted as `Multilayer` / `RepeatingMultilayer` constructor
  arguments and serialized from the current graph state, so the ties are
  rebuilt on `from_dict`.
- Structural parameter paths (`Project.parameter_path` /
  `Project.resolve_parameter_path`) address parameters stably across
  save/load.
- `Parameter.bounds = (lo, hi)` assignments in tutorials, notebooks and
  integration tests migrated to `.min` / `.max`.
- ORSO model loading now uses the parsed `SampleModel` as-is, so named
  materials, sub-stacks and composits are no longer dropped (named
  materials previously read back with SLD 0).

## Polarization

All four polarization channels (pp, pm, mp, mm) are now available from
the refl1d calculator. Previously only the non-spin-flip pp channel was
returned.

- New `LayerMagnetism` sample element. `Layer` takes an optional
  `magnetism` with fittable, serialized `Parameter`s `rho_m` (magnetic
  SLD) and `theta_m` (in-plane moment angle). Adding a magnetic layer
  turns on `include_magnetism` on the calculator, or raises
  `NotImplementedError` if the backend cannot do magnetism. Removing the
  last magnetic layer turns it off again. `Model.has_magnetism`,
  `CalculatorBase.supports_magnetism` and
  `Project.calculator_supports_magnetism` report the current state.
- New `PolarizedDataSet` groups per-channel `DataSet1D` objects (one
  file per channel; NSF experiments use 'pp'/'mm' only, spin-flip
  channels are optional) into one experiment that shares a single model.
  `Project.load_polarized_experiment(paths)` loads from an explicit
  channel-to-file mapping.
  `Project.suggest_polarized_channel_assignment(paths)` fills that
  mapping from the ORSO header polarization (`pp`/`mm`/`pm`/`mp` only).
  Partially analysed observables such as `po`/`mo` (channel sums) and
  `op`/`om`/`unpolarized` are left for the user. For plain text files
  the mapping comes from filename tokens (`_uu`/`_up`/`_pp` → pp,
  `_dd`/`_down`/`_mm` → mm, `_ud`/`_pm` → pm, `_du`/`_mp` → mp).
- Experiment and model accessors are channel-aware.
  `Project.experimental_data_for_model_at_index(index, channel=...)`
  returns the `DataSet1D` of one spin channel. `channel=None` (the
  default) still returns the stored experiment.
  `Project.model_data_for_model_at_index(index, q_range, channel=...)`
  calculates one spin cross-section.
  `Project.experiment_is_polarized_at_index(index)` and
  `Project.experiment_channels_at_index(index)` report the polarization
  state. A channel that was not measured raises `KeyError`. An unknown
  channel, or any channel on an unpolarized experiment, raises
  `ValueError`.
- Summary/report figures now plot one measured series per spin channel
  of a polarized experiment, each in its channel colour, plus the
  matching calculated cross-section. Channels that cannot be calculated
  (for example spin-flip on a non-magnetic model) are shown without a
  calculated overlay. Previously a polarized experiment made the report
  figures fail on `PolarizedDataSet.x`.
- The summary experiments table lists one row per spin channel of a
  polarized experiment, named `<experiment> (<channel>)`. It previously
  raised
  `AttributeError: 'PolarizedDataSet' object has no attribute 'x'` and
  crashed anything that read the summary while a polarized experiment
  was loaded.
- New `Project.calculators_supporting_magnetism` lists the available
  calculators that can model magnetic samples, without switching the
  active one. `Project.models_have_magnetism` reports whether any model
  has a magnetic layer. Use these to pick a suitable engine, or to
  refuse one that cannot carry the sample's magnetism, instead of
  hitting an error inside the binding.
- New `Project.magnetic_sld_data_for_model_at_index(index)` returns the
  depth profiles of a magnetic model as `DataSet1D`s keyed `'sld'`,
  `'rho_m'`, `'theta_m'`, `'spin_up'` and `'spin_down'`. The last two
  are the potentials each spin state sees, rho +/- rho_m\*cos(theta_m -
  A). The guide-field angle A is the new module constant
  `GUIDE_FIELD_ANGLE` (270 degrees, refl1d's default and the only value
  the library can currently model). A non-magnetic model raises
  `ValueError`. `Project.model_has_magnetism_at_index(index)` reports
  whether the model is magnetic.
- The magnetic depth profile is now built by smoothing the two in-plane
  components of the moment and converting back, rather than smoothing
  magnitude and angle separately as refl1d does channel by channel. At
  an interface where moments differ by a few degrees across 0/360, the
  smoothed _angle_ used to take the long way around the circle, pass
  through the guide-field direction, and report the full moment as
  longitudinal. That produced a spurious spin-up/spin-down splitting
  exactly at the interface (a 2-degree difference gave the full 2*rho_m
  splitting; it is now the correct ~0.02*rho_m). Collinear samples are
  unaffected. The reported `theta_m` profile is restricted to depths
  that carry a moment (the angle of a zero-length vector is arbitrary)
  and is made continuous within each magnetic region. A profile turning
  from 359 to 1 degree is a 2 degree turn; the wrapped values would plot
  as a full sweep. If the installed refl1d does not expose the microslab
  data the component-safe profile needs, the calculator now raises
  `NotImplementedError` instead of falling back to the angle-smoothed
  profile.
- New `Project.spin_asymmetry_for_experiment_at_index(index)` returns
  the measured spin asymmetry (R++ - R--)/(R++ + R--) of a polarized
  experiment, the matching model curve when the model is magnetic, and
  the number of points dropped. `ye` holds the SA **variance**, as
  everywhere else in the library. Channels measured on different q grids
  are interpolated onto the pp grid (values with the linear weights,
  variances with their squares) only inside the q range both channels
  cover. Outside that range `np.interp` would clamp to the edge value.
  Dropped points are reported as `out_of_overlap_points`. Points where
  R++ + R-- is not above `SPIN_ASYMMETRY_SIGNIFICANCE` (3) times its own
  uncertainty are also dropped. A second, uncertainty-independent guard
  drops points whose denominator is non-positive or smaller than
  `SPIN_ASYMMETRY_CANCELLATION_FRACTION` (1e-3) of |R++| + |R--|.
  Without it, a file with no uncertainties (two columns, or a malformed
  uncertainty array) had no guard, and background-subtracted data could
  put values of +/-1e3 on the axis. Points with a non-finite
  reflectivity or a negative/non-finite variance are dropped rather than
  treated as having no uncertainty. Dropped points are reported by
  reason (`low_significance_points`, `small_denominator_points`,
  `invalid_points`).
- Both channels of a spin asymmetry are validated before use. Empty,
  length-mismatched, non-finite or duplicated q grids are rejected, and
  `experiment_supports_spin_asymmetry_at_index` reports False for them.
  A descending grid is sorted before pairing; `np.interp` silently
  returns nonsense for one.
  `Project.experiment_supports_spin_asymmetry_at_index(index)` reports
  whether both non-spin-flip channels were measured.
- New `calculate_channel(q, model, channel)` on the wrapper (and
  `reflectivity_profile_channel` on the calculator,
  `fit_func_for_channel` on `CalculatorFactory`) evaluates one explicit
  spin channel without touching the global `polarization_channel` state.
- New `MultiFitter.for_experiments(experiments)` builds a fitter with
  one fit function per dataset (one per measured spin channel for a
  polarized experiment, one for an ordinary one) across any number of
  experiments and models, and returns without running the fit.
  `fit_datasets` and `fit_channels` give the flat dataset list in
  fit-function order, so an application can prepare the data arrays and
  drive `easy_science_multi_fitter.fit(...)` from a worker thread.
- New `MultiFitter.record_fit_results(results)` adopts results from such
  a caller-driven fit, so `chi2` and `reduced_chi` describe it instead
  of reporting that no fit was performed. The classical metrics need the
  original data arrays and stay None.
- `rho_m` now takes part in the project's default-limit policy. It is
  created with `default_limits_pending`, and
  `Project._sync_parameter_states` gives it the shared SLD window (-1
  to 10) unless an explicit `Parameter` with its own bounds was passed.
  `theta_m` keeps its explicit 0-360 bounds. Previously both stayed
  unbounded.
- New `MultiFitter.fit_polarized(data)` fits all measured channels of a
  `PolarizedDataSet` simultaneously against the shared model: one fit
  function per channel, common structural parameters, magnetic
  parameters constrained by all channels at once. Returns per-channel
  `FitResults`.
- The refl1d wrapper now caches the four polarized cross-sections per
  model state and (q, dq) grid. They come from a single kernel
  evaluation, so a simultaneous N-channel fit costs about one evaluation
  per iteration instead of N.

- New `polarized_reflectivity_profiles(x_array, model_id)` on the
  calculator (and on `CalculatorFactory`) returns the reflectivity of
  all four spin channels in one calculation as a dictionary keyed
  `'pp'`, `'pm'`, `'mp'`, `'mm'` (in that order). Requires
  `include_magnetism = True`.
- New `polarization_channel` property (accepts
  `'pp'`/`'pm'`/`'mp'`/`'mm'` or the new `PolarizationChannel` enum)
  selects which channel `reflectity_profile` (and therefore fitting)
  returns, so fits can target spin-flip or mm data. Default `'pp'`;
  disabling magnetism resets it to `'pp'`. The channel belongs to the
  currently active calculator instance, not to a model or dataset. It
  affects every subsequent calculation with that calculator.
  `interface.switch(...)` constructs a fresh calculator and resets both
  this and `include_magnetism`.
- New `magnetic_sld_profile(model_id)` on the calculator (and on
  `CalculatorFactory`) returns the nuclear and magnetic scattering
  length density profiles as a tuple `z`, `sld(z)`, `rhoM(z)` (magnetic
  SLD) and `thetaM(z)` (magnetic angle). Requires
  `include_magnetism = True`; refl1d only.
- Magnetic calculations now always build all four refl1d cross-sections,
  so they may take somewhat longer than before. pp results are
  unchanged.
- Bug fix: `include_magnetism = True` on a refnx-backed calculator now
  raises `NotImplementedError`. Previously it was silently accepted (the
  guard sat on a property the calculator never called) even though refnx
  magnetism is not supported.
- Bug fix (pre-existing): disabling magnetism after layers were created
  with it enabled used to leave refl1d `Magnetism` objects on the slabs,
  and a later unpolarized calculation raised `AttributeError` inside
  refl1d. Disabling magnetism now strips the magnetic state from
  existing layers. Magnetic parameters (`rhoM`/`thetaM`) are kept in a
  per-layer store inside the wrapper, so they survive a
  disable/re-enable cycle and are re-attached when magnetism is enabled
  again. `update_layer` also accepts the magnetism keys one at a time.

## ORSO file handling

- Binary ORSO (`.orb`, NeXus/HDF5) files are read and written alongside
  `.ort` text files. The format is detected from the ORSO banner line or
  the HDF5 magic bytes, not the file extension, so a file that carries
  the banner but fails to parse now raises instead of being silently
  re-read as plain text (which dropped the whole header, polarization
  included). `.orb` support needs `h5py`, available as the new `orb`
  extra; `orsopy` is pinned to `>=1.2`.
- New ORSO export. `Project.save_experiment_as_orso(path, index=None)`
  writes an experiment (`.ort` or `.orb`), with the model, when set,
  serialized as `data_source.sample.model`. Backed by the new
  `save_orso_experiment`, `orso_datasets_from_experiment` and
  `sample_to_orso_model` in `easyreflectometry.orso_utils`. A polarized
  experiment becomes one file with one `data_set:` block per spin
  channel. `Model.as_orso` now returns the ORSO model-language
  dictionary (slab representation) rather than the internal `as_dict`.
- Repeating multilayers survive a round trip. Loading resolves the ORSO
  stack with `resolve_stack()` instead of flattening it, so a sub-stack
  keeps its repetition count and comes back as a `RepeatingMultilayer`;
  export writes it with the inline `N ( ... )` stack syntax.
- Units declared in the file are honoured: `Qz` in `1/nm`, lengths in
  `nm` (the ORSO default) and SLDs in `1/nm^2` are converted on load,
  instead of being read as angstrom-based numbers.
- Resolution and error columns are read more carefully. A column
  declared `value_is: FWHM` is converted to sigma on load, `nan` entries
  in `sQz` are filled by interpolating over the valid points, and
  partially missing error columns warn rather than propagating `nan`
  into a fit. Stored `Pointwise` resolutions remain variances, so saved
  projects round-trip without migration.
- `Project.load_polarized_experiment_from_file(path)` loads a polarized
  experiment from a single multi-dataset ORSO file, classifying each
  `data_set:` block by its own `instrument_settings.polarization`
  header. Only `pp/pm/mp/mm` are mapped; a file with an unmappable or
  duplicated channel raises rather than guessing. Supported by the new
  `channel_from_orso_polarization` and
  `detect_polarization_channels_per_dataset` in
  `easyreflectometry.data`.
- New `easyreflectometry.data.dataset_from_datagroup` builds a
  `DataSet1D` from one dataset of an already-loaded `DataGroup`, and
  keeps the parsed ORSO header on the dataset as `orso_header` so
  exporters can reuse the original provenance. `load_as_dataset` and the
  project loaders accept a pre-loaded `DataGroup`, so importing a file
  no longer parses it three or four times.

## Documentation

- The documentation is now MkDocs (Material) only. The legacy Sphinx
  tree (`docs/src`, `docs/Makefile`, `docs/make.bat`) and the
  tag-triggered `documentation-build.yml` workflow have been removed;
  the site is built and deployed by `docs.yml` from `docs/mkdocs.yml`.
- New tutorials wired into the navigation: _Constraints & Inequalities_
  and _Bayesian Fitting_.
- New API reference pages for constraints, inequality constraints,
  Bayesian analysis, calculators, parameter limits, `LayerMagnetism`,
  ORSO, summary and plotting.

# Version 1.7.0 (1 Aug 2026)

Restored the measured per-point resolution on data load (issue #368).

- Loading data through `Project` (`load_new_experiment`,
  `load_experiment_for_model_at_index`,
  `load_all_experiments_from_file`) again sets a `Pointwise` resolution
  function when the file carries per-point q-resolution (an sQz column
  in `.ort` files, or a 4th column in text files). Since PR #293 the
  loaders discarded this data and always applied a flat
  `PercentageFwhm(5.0)` — a temporary workaround that never got
  reverted. Fits of such data were smeared with 5% FWHM regardless of
  what the instrument delivered and should be re-run.
- Files without q-resolution data keep the 5% FWHM default. The pre-#293
  fallback that built a `LinearSpline` from the _reflectivity_ error
  (`sqrt(ye)`) was not restored: a reflectivity uncertainty is not a
  q-width, and that branch produced effectively zero smearing.
- Known limitation (pre-existing): the resolution function lives on the
  model, so when several experiments share one model the last-loaded
  dataset's resolution wins.

Fixed inconsistent interpretation of vector resolution functions between
the refnx and refl1d engines (issue #367).

- **Reflectivity results change for two engine / resolution
  combinations.** `LinearSpline` on refl1d previously **over-smeared by
  a factor of 2.355** (its FWHM widths were passed to refl1d's
  `probe.dQ`, which expects sigma). `Pointwise` on refnx previously
  **under-smeared by the same factor** (its sigma widths were passed to
  refnx's `x_err`, which expects FWHM). Both are now correct. Fits and
  simulations that used either combination will produce different —
  previously wrong — results and should be re-run. `PercentageFwhm` on
  either engine, `LinearSpline` on refnx, and `Pointwise` on refl1d are
  numerically unchanged.
- `ResolutionFunction.smearing()` now returns **sigma** (the Gaussian
  standard deviation) for every subclass; each engine wrapper converts
  to its backend's convention. This is a behavioural change to a public
  method. Most visibly, `PercentageFwhm.smearing(q)` used to return the
  _percentage_ itself (e.g. `5.0`) and now returns an absolute sigma
  (e.g. `0.00212` at `q=0.1`); `LinearSpline.smearing(q)` returns its
  `fwhm_values` divided by `2*sqrt(2*ln2)`. Callers relying on the old
  values need to convert. The new `SIGMA_TO_FWHM` constant is exported
  from `easyreflectometry.model.resolution_functions`.
- Constructors are **unchanged**: `PercentageFwhm(5)` still means 5%
  FWHM and `LinearSpline(q, fwhm_values)` still takes FWHM. Only the
  `smearing()` output convention moved, so existing model-building code
  needs no edits.
- `PercentageFwhm.smearing(q)` given a scalar `q` now returns a 0-d
  numpy scalar rather than a shape-`(1,)` array, matching
  `LinearSpline`. `smearing(0.1)[0]` therefore raises `IndexError` where
  it previously returned a value.

Migrated sample / model classes off the deprecated `easyscience.ObjBase`
and `easyscience.CollectionBase` pipeline.

- `BaseCore` is now built on `ModelBase`; `BaseCollection` on
  `EasyList`. `Model`, `Material`, `Layer`, `MaterialMixture`,
  `MaterialSolvated`, `LayerAreaPerMolecule`, `Multilayer`,
  `RepeatingMultilayer`, `GradientLayer`, `Bilayer`, `SurfactantLayer`,
  `BaseAssembly`, `LayerCollection`, `MaterialCollection`, `Sample`, and
  `ModelCollection` were all rewritten to use the new bases.
- Properties returning a `Parameter` (`Material.sld`-style) now expose
  the `Parameter` object directly across all sample classes, replacing
  the inconsistent legacy behaviour where `MaterialMixture.fraction`,
  `MaterialSolvated.solvent_fraction`,
  `LayerAreaPerMolecule.area_per_molecule`, and
  `LayerAreaPerMolecule.solvent_fraction` returned `float`. Read the
  value via `.value` (e.g. `material_mixture.fraction.value`). Setters
  still accept a float. `MaterialMixture.sld` / `MaterialMixture.isld`
  remain `float` — they are derived via constraints, not constructor
  arguments.
- `BaseCollection.remove(index)` (the legacy index-based helper) renamed
  to `remove_at(index)`. The standard `MutableSequence.remove(value)` is
  now inherited unmodified.
- Project files saved by previous versions cannot be read.
  `Project.as_dict` writes `file_format=2`; `Project.from_dict` raises a
  clear `ValueError` on missing or unsupported markers.
- `model.get_parameters()` / `collection.get_parameters()` still work
  (kept as compatibility shims) but new code should use
  `get_all_parameters()`.
- No more `DeprecationWarning` from `easyscience.ObjBase` /
  `CollectionBase` on construction of any sample / model object.

# Version 1.6.0 (1 May 2026)

Add Mighell-based handling of non-positive-variance points in fitting
(issue #256). Non-positive-variance data points are no longer forcibly
discarded; instead, a hybrid objective applies a Mighell substitution
for non-positive-variance points while using standard weighted least
squares for the rest. The previous masking behavior is available via
`objective='legacy_mask'`. New `objective` parameter on `MultiFitter`,
`fit()`, and `fit_single_data_set_1d()`.

# Version 1.3.3 (17 June 2025)

Added Chi^2 and fit status to fitting results. Added explicit dependency
on bumps version.
