# Constraints

EasyReflectometry offers three kinds of constraints between model
parameters: equality constraints, derived read-only parameters and
[inequality constraints](inequality_constraints.md). The
[Constraints tutorial](../tutorials/advancedfitting/constraints.ipynb)
walks through all three on a worked example.

## Equality constraints (dependencies)

A parameter can be tied to an arbitrary expression of other parameters.
It then leaves the set of free fit parameters and follows the
expression.

```python
from easyreflectometry import constrain
from easyreflectometry import constrain_equal
from easyreflectometry import unconstrain

constrain_equal(layer_b.roughness, to=layer_a.roughness)
constrain(layer_b.thickness, '2 * t', t=layer_a.thickness)
unconstrain(layer_b.thickness)
```

Constraints survive `Project` save/load: the expression and the
structural paths of the parameters it refers to are stored with the
project, and the graph is rebuilt when it is loaded. This covers the
helpers above; a dependency created by calling `make_dependent_on`
directly is not recorded.

## Derived (read-only) parameters

A _derived parameter_ is a dependent parameter that belongs to no layer:
a live calculation that can be shown or referenced from an equality
constraint.

```python
from easyreflectometry import constrain_to_sum
from easyreflectometry import derived_parameter

total = derived_parameter('total', 'a + b', a=layer_a.thickness, b=layer_b.thickness)
# keep the film thickness fixed at 120 Å while the split is fitted
constrain_to_sum(layer_b.thickness, [layer_a.thickness, layer_b.thickness], total=120.0)
```

!!! warning

    A standalone derived parameter is **session-only**: it has no
    structural path, so it cannot be named in an inequality constraint,
    and a project whose equality constraints depend on one cannot be
    saved (`Project.as_dict` raises). Numeric totals (as above) are
    fine — they are embedded by value.

For a derived value that persists and can be used in inequalities, use
one owned by the model: every [`Model`](model.md) exposes
`total_thickness`, the summed thickness of the layers between the
superphase and the subphase, re-derived whenever the layer structure
changes.

## Guarding a sum remainder

The parameter tied by `constrain_to_sum` absorbs whatever the others
leave over, so on its own the constraint lets a fit push the partners
past the total and drive the remainder negative — a layer of negative
thickness. `clamp_sum_partners` narrows each partner's `max` to the
headroom it actually leaves, sharing the slack in proportion to the
current values, and `restore_sum_partners` hands the original maxima
back when the constraint is released.

```python
from easyreflectometry import clamp_sum_partners
from easyreflectometry import is_constrained_to_sum
from easyreflectometry import restore_sum_partners

clamp_sum_partners([layer_a.thickness], remainder=layer_b.thickness.value)
is_constrained_to_sum(layer_b.thickness)  # True, also after a project reload

unconstrain(layer_b.thickness)
restore_sum_partners([layer_a.thickness])
```

The narrowed maxima are persisted by structural path, so removing the
constraint after a save/load still gives the original bounds back.

::: easyreflectometry.constraints
