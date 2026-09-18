# Inequality Constraints

Cross-parameter inequalities such as `t_head < t_tail` or
`t1 + t2 <= total` are not dependencies: no parameter is removed from
the fit. They are declared as `InequalitySpec` objects on the project
and enforced by the **BUMPS** engines (the `Bumps*` minimizers and the
DREAM sampler) as penalties on the fit problem. LMFit and DFO-LS cannot
enforce them, and `fit` raises `ValueError` in that case rather than
silently dropping the physics.

```python
from easyscience.fitting import AvailableMinimizers

from easyreflectometry import InequalitySpec

project.minimizer = AvailableMinimizers.Bumps
t_a = project.parameter_path(layer_a.thickness)  # 'models/0/sample/1/layers/0/thickness'
t_b = project.parameter_path(layer_b.thickness)
project.add_inequality_constraint(InequalitySpec('a', '<', 'b', {'a': t_a}, {'b': t_b}, name='order'))
project.add_inequality_constraint(InequalitySpec('a + b', '<', '90', {'a': t_a, 'b': t_b}, {}))

project.violated_inequality_constraints()  # check the start point first
project.fitter.fit_single_data_set_1d(dataset)  # penalties applied automatically
```

Parameters are referenced by _structural path_ (see
`Project.parameter_path`) so the constraints are saved with the project.
While a constraint is violated BUMPS skips the model evaluation and adds
a penalty growing with the violation, steering the optimizer back into
the feasible region; the `Bumps_lm` method spreads the penalty over the
residuals instead and enforces inequalities more weakly.

Both sides of a spec are unit-checked when it is registered; a mismatch
raises `UnitError`, a subclass of `ValueError`.

::: easyreflectometry.inequality_constraints
