# Provenance of third-party test data

## `e361r.txt`

- **Source:** refnx 0.1.63, `refnx/reflect/tests/e361r.txt`, copied
  unchanged from the installed package (also shipped as
  `refnx/analysis/tests/e361r.txt`).
- **Licence:** BSD-3-Clause (refnx).
- **Content:** 99 points, three columns: Q (Å⁻¹), R, dR (standard
  deviation). No resolution column.
- **Model refnx uses with it** (`refnx/reflect/tests/test_reflect.py`):
  `si | sio2(10, 4) | polymer(200, 3) | d2o(0, 3)` with `dq = 5` % FWHM;
  free: scale [0.1, 2], background [0, 5e-5], D2O SLD [6, 6.36], SiO₂
  thickness [5, 20], polymer thickness [100, 220], polymer SLD [0.2,
  1.5]. refnx calls it "an older dataset, but well characterised". It is
  a polymer film on silicon, not a lipid bilayer.
- **Used by:** `tests/integration/convergence/`, as a representative
  fixture for the LMFit convergence investigation. It is not the case
  the original complaint was made on.

## `example.ort`

- **Source:** the refnx documentation example (ORSO header: "Example
  data file from refnx docs", instrument Platypus, owner Andrew Nelson).
- **Content:** 408 points, four columns including sQz, so a loaded
  dataset carries a `Pointwise` resolution; fitting it as an experiment
  exercises per-dataset resolution handling.

## `validation_test4.dat`

- **Source:** the `reflectivity/analysis` validation suite,
  `validation/test/unpolarised/data/test4.dat`
  (https://github.com/reflectivity/analysis). Copied unchanged from the
  `stateless` branch's `tests/calculators/refnx/_validation_data.py`.
- **Licence:** see that repository.
- **Content:** 101 points, four columns: Q (Å⁻¹), R, dR, dQ (sigma,
  Å⁻¹). Calculated, not measured: an independent reference, not a
  recording of this library.
- **Model:** the suite's four-layer test0/test4 model (SLD 2.07 | 3.45 +
  0.1i, 100 Å, σ 3 | 5.0 + 0.01i, 200 Å, σ 1 | 6.0, σ 5), at 5 %
  constant resolution or the pointwise dQ column.
- **Used by:** `tests/calculators/test_validation_suite.py`, against
  both engines.
