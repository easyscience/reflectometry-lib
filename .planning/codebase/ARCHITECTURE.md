# Architecture

**Analysis Date:** 2026-10-15 **Branch analysed:** `develop` @
`66adaf54`

## Pattern Overview

**Overall:** Layered library with two facades. `Project` is the
stateful, GUI-oriented facade (experiments, models, persistence,
constraints, contrasts); `MultiFitter` is the fitting facade that turns
datasets into a `PreparedFit` and delegates execution to EasyScience's
`MultiFitter` / `Sampler`, which drive LMFit, BUMPS or DFO. Reflectivity
is computed by pluggable engine wrappers (refnx, refl1d) behind
`CalculatorFactory`.

**Key Characteristics:**

- Prepare / execute / record split: `Project.prepare_fit` →
  `PreparedFit` (immutable snapshot of settings, arrays, bound fit
  functions, configured core fitter) → `execute()` or a GUI worker →
  `Project.record_fit` → `FitRun`
- Resolution follows the dataset, not the calculator: each
  `DataSet1D.resolution_function` is bound into its fit function;
  `CalculatorBase` keeps one resolution per model id
- Parameters are EasyScience `Parameter`s; all coupling (conformal
  thickness, contrasts, user constraints, sum constraints) is expressed
  as parameter dependencies and persisted by structural path
- Inequality constraints are enforced by patching the BUMPS driver in
  the core's namespace, because the core offers no hook
- Shared objects (one material in two layers, one assembly in two
  models) serialize once through `sample/references.py`
  (`file_format` 3)

## Layers

**Project Facade:**

- Purpose: Hold the session state an application needs and expose
  index-based accessors for plots and tables
- Location: `src/easyreflectometry/project.py` (`Project`,
  `ModelRemoval`, spin-asymmetry helpers)
- Contains: experiments `Dict[int, DataSet1D | PolarizedDataSet]`,
  `ModelCollection`, `MaterialCollection` palette, `FitSettings`,
  inequality specs, link records, `last_fit`, load/save, ORSO import,
  contrasts, polarized loading, `spin_asymmetry_for_experiment_at_index`
- Depends on: every other package in the library
- Used by: EasyReflectometry GUI (ERA app), tutorials, `Summary`

**Fitting Layer:**

- Purpose: Zero-variance handling, per-dataset resolution binding,
  settings snapshot, constraint enforcement, goodness-of-fit metrics
- Location: `src/easyreflectometry/fitting.py`, `fit_settings.py`,
  `_bumps_constraints.py`, `inequality_constraints.py`
- Contains: `MultiFitter` (`fit`, `fit_single_data_set_1d`,
  `fit_polarized`, `for_experiments`, `prepare`, `mcmc_sample`,
  `switch_minimizer`, chi-square properties), `PreparedFit`, `FitRun`,
  `_ConstrainedEasyScienceMultiFitter`, `FitSettings`,
  `MinimizerOption`, `InequalitySpec`, `build_constraints_factory`
- Depends on: `easyscience.fitting` (`MultiFitter`, `Sampler`,
  `AvailableMinimizers`, `FitResults`), `asteval`, `scipp`

**Bayesian Analysis Layer:**

- Purpose: Post-hoc posterior analysis and chain persistence
- Location: `src/easyreflectometry/analysis/bayesian.py`
- Contains: `PosteriorResults` (summary, corner, trace, distribution,
  credible*interval, gelman_rubin, save), `posterior_predictive*\*`,
  `save_posterior`/`load_posterior` with fingerprinted sidecar
- Depends on: plotly (required), arviz (optional), numpy

**Model / Sample Layer:**

- Purpose: Physical description of the sample and measurement
- Location: `src/easyreflectometry/model/`,
  `src/easyreflectometry/sample/`
- Contains: `Model` (scale, background, resolution, `total_thickness`
  derived parameter), `Sample` and assemblies/layers/materials,
  `LayerMagnetism`, `MaterialDensity.sld_coupled`, `volume_fraction`,
  `paths`, `references`
- Depends on: `easyscience.base_classes`, `easyscience.variable`,
  periodictable (via `special/`)

**Constraints and Contrasts Layer:**

- Purpose: User-level coupling between parameters and between models
- Location: `src/easyreflectometry/constraints.py`, `contrasts.py`
- Contains: `constrain*`, `derived_parameter`, `clamp_sum_partners`;
  `derive_contrast`, `ReplaceMaterial`, `ReplaceFormula`, `plan_link` /
  `LinkPlan` / `LinkRecord`, `tie_corresponding`, `snapshot` / `restore`
- Depends on: EasyScience dependency mechanism; `USER_CONSTRAINT_FLAG`
  marks constraints the project should persist

**Data Layer:**

- Purpose: Load, merge and describe measured data
- Location: `src/easyreflectometry/data/`, `orso_utils.py`
- Contains: `DataSet1D`, `PolarizedDataSet`, `load`, `load_as_dataset`,
  `merge_datasets`, `resolution_from_dataset`, channel detection, ORSO
  read/write and model-language conversion

**Calculator Layer:**

- Purpose: Pluggable reflectivity engines
- Location: `src/easyreflectometry/calculators/`
- Contains: `CalculatorBase` (item lifecycle via `ItemContainer`s,
  per-model resolution, magnetism switches), `WrapperBase` (abstract
  engine API incl. `calculate_channel`, `calculate_polarized`,
  `magnetic_sld_profile`), `CalculatorFactory` (`fit_func`,
  `fit_func_for(channel, resolution_function)`), `PolarizationChannel`
- Implementations: `refnx/` (unpolarized), `refl1d/` (magnetism,
  polarized probes, oversampled resolution)

**EasyScience Core (external, pinned 2.5.1):**

- `Fitter` / `MultiFitter` (reshaping of multi-dataset arrays, minimizer
  dispatch), `Sampler` / `SamplingResults` (DREAM chain, `extend`,
  `save`, `load_state`), minimizer wrappers, `InterfaceFactoryTemplate`

## Data Flow

**Classical Fit (project path):**

1. GUI/user calls
   `Project.prepare_fit(experiments=None, objective=None, skip_invalid=False)`
2. `_fit_scope` selects experiments with `include_in_fit` and a model;
   `_fitter_for` builds `MultiFitter.for_experiments(...)` with a deep
   copy of `fit_settings` and a weak `constraints_factory_provider`
3. `MultiFitter.prepare` applies the objective (`hybrid` default:
   Mighell for zero-variance points, WLS otherwise), binds each
   dataset's fit function to its own resolution
   (`_resolution_for_fitted_points` for `legacy_mask`), collects
   `added_roots` (free parameters of left-out models that fitted
   parameters follow), builds the core fitter via `_core_fitter_for` +
   `FitSettings.configure`, checks preconditions (finite bounds for
   engines that need them, warns on starts at bounds)
4. Caller runs `prepared.execute(**kwargs)` or `core_fitter.fit(...)` in
   a thread; the `_ConstrainedEasyScienceMultiFitter.fit` property wraps
   the call in `_bumps_constraints.applied(factory)`
5. `Project.record_fit(prepared, results, status)` computes classical
   metrics from the measured arrays (`prepared.finalize`), stores a
   `FitRun` on `last_fit` and pushes results to `project.fitter`

**Classical Fit (library path):**

1. `MultiFitter(model, ...)` and `fitter.fit(data: sc.DataGroup)`
2. `_datagroup_inputs` pairs curve `Qz_<n>` with model/dataset `n` in
   numeric order
3. `prepare` → `execute` → results written back into the `DataGroup`
   (model curves, SLD profiles, pooled chi-square)

**Polarized Fit:**

- `fit_polarized(polarized_dataset)` or `for_experiments([...])`: one
  fit function per spin channel, all against one model, each bound to
  its channel via `CalculatorFactory.fit_func_for(channel, resolution)`.
  Structural parameters shared, magnetic parameters constrained by all
  channels.

**Bayesian MCMC:**

1. `Project.prepare_sample()` → `for_experiments` fitter with settings
   snapshot; or `fitter.mcmc_sample(data)` directly
2. `mcmc_sample` requires a BUMPS minimizer
   (`minimizer.package == 'bumps'`), refuses all-zero variances unless
   `objective='mighell'`, calls `prepare(action='sampling')`
3. Builds `Sampler(core_fitter, x, y, weights)` inside the constraints
   context, patches `sampler.extend` to re-enter it, keeps the sampler
   on `fitter.sampler`, runs
   `sampler.sample(samples, burn, thin, population, sampler_kwargs, progress_callback, abort_test)`
4. Returns `{'draws', 'param_names', 'state', 'logp'}`; user wraps in
   `PosteriorResults` or calls `save_posterior`

**Persistence:**

- `Project.as_dict` writes models, materials in order, experiments by
  index and name with per-dataset resolution, fit settings, inequality
  specs, user-constraint records by path, contrast provenance, links.
  `references.writing()` replaces repeated objects with `@ref`.
  `_needs_new_format` bumps `file_format` to 3 when older readers would
  misread the file.
- `from_dict` rebuilds shared objects (`references.reading`), restores
  constraints by path, reports problems in `load_report`.

**State Management:**

- `Project._sync_parameter_states` disables physically meaningless
  parameters (superphase thickness/roughness, subphase thickness) and
  applies default limits to enabled parameters created from project
  defaults
- `PreparedFit` owns copies of the measured arrays; model parameters and
  constraints stay live because they are what the fit changes
- Parameter values saved/restored around posterior predictive evaluation
  (`_save_parameter_state` / `_restore_parameter_state`)

## Key Abstractions

**`PreparedFit` (fitting.py):** Snapshot pattern. Everything a run
needs, fixed at prepare time; owns copies of measured arrays, the
configured core fitter and both full-resolution (`curve_funcs`) and
fitted-point (`fit_funcs`) evaluators.

**`FitRun` (fitting.py):** Frozen record of a run (status, minimizer,
objective, inputs by experiment key, pooled and per-dataset statistics).
Outlives model changes; `remapped()` follows experiment re-keying.

**`FitSettings` (fit_settings.py):** Serializable configuration
(minimizer, mode, objective, tolerance, budget, per-minimizer options
validated by `MinimizerOption` schemas). `configure(fitter)` and
`engine_kwargs()` are the two consumers.

**`_ConstrainedEasyScienceMultiFitter` / `_bumps_constraints`:**
Property override plus monkeypatch. The subclass overrides the core's
read-only `fit` property so the raw `easy_science_multi_fitter.fit(...)`
path resolves the owner's constraints provider in the calling thread and
consumes an explicit `constraints_factory` keyword (the core would
swallow it via `**kwargs`). `install()`, run at import of
`_bumps_constraints`, wraps `minimizer_bumps.FitDriver` so each
`FitProblem` gets `constraints = factory(pars)` before the driver runs.
Non-BUMPS engines raise `NON_BUMPS_ERROR`.

**`InequalitySpec` (inequality_constraints.py):** Declarative
`lhs op rhs` over parameter paths, evaluated with asteval in a
restricted symbol table; `build_constraints_factory` returns BUMPS
penalty callables.

**Contrasts (`contrasts.py`):** A contrast is an ordinary `Model`
sharing structure with a reference by identity or by `constrain_equal`.
`derive_contrast` copies touched assemblies, shares the rest and ties
parameters; `plan_link` / `apply_link` / `unlink` manage equalities
between two models of one layout with full state snapshots.

**`CalculatorBase` / `WrapperBase` / `CalculatorFactory`:** Template +
Adapter + Factory. Items are claimed per calculator (`_claim`),
per-model resolution registry, magnetism toggles, channel-bound fit
functions.

**`references` (sample/references.py):** Context-managed shared-object
serialization (`writing()`, `reading(report)`, `suspended()`), with
`prune`, `rebuild` and `register` for load-time reconciliation.

**`paths.walk` (sample/paths.py):** Structural addressing of every
parameter; the stable identity used by constraints and persistence
instead of EasyScience unique names.

## Entry Points

- `Project.prepare_fit` / `prepare_sample` / `record_fit` - GUI fit and
  sampling
- `Project.load_experiment_from_files`, `load_polarized_experiment*`,
  `load_orso_file`, `save_as_json`, `load_from_json`
- `Project.add_contrast`, `plan_link`, `apply_link`, `detach`
- `MultiFitter.fit`, `fit_polarized`, `mcmc_sample`
- `Summary(project).save_html_summary` / `save_pdf_summary`
- `python -m easyreflectometry.main` prints available calculators (no
  console script registered)

## Error Handling

**Strategy:** Fail loudly at preparation time; warn for recoverable data
issues; refuse silent degradation.

**Patterns:**

- `FitScopeError` (nothing to fit / experiment without model),
  `FitPreconditionError` (unusable bounds for the engine), `ValueError`
  for invalid settings or empty datasets after masking
- `RuntimeError` when sampling without a BUMPS minimizer
- `ValueError(NON_BUMPS_ERROR)` when inequality constraints meet a
  non-BUMPS engine
- `UnitError` (subclass of `ValueError`) for inequality unit mismatches
- `FileExistsError` / `FileNotFoundError` / `OSError` / `ValueError`
  from project persistence (no printing)
- `warnings.warn` for zero-variance handling, ORSO column problems,
  negative volume fractions, unreadable dependencies on load; load
  issues also collected in `Project.load_report`
- `try/finally` restores parameter state around posterior predictive;
  `Project.save_as_json` writes atomically

## Cross-Cutting Concerns

**Logging:** `logging.getLogger(__name__)` in a handful of modules; no
handler configured. `warnings` is the main user-facing channel.

**Validation:** `FitSettings.validate`, `MinimizerOption.validate`,
`InequalitySpec.validate_syntax` + `check_units`, ORSO column
validation, `_check_fit_preconditions`.

**Threading:** The constraints factory lives in a `contextvars`
variable, resolved in the calling thread; `PreparedFit` owns array
copies so a worker thread is isolated from later edits. Resolution for
one-off curves is bound to the evaluator rather than swapped on the
model to avoid racing a running fit.

**Serialization:** EasyScience serializer plus library extensions:
`references` for shared objects, structural paths for constraints, per
dataset resolution entries, `file_format` negotiation. Posterior chains
have their own sidecar format.

**Authentication:** Not applicable.

---

_Architecture analysis: 2026-10-15 (develop @ 66adaf54)_
