# PHNN implementation boundary

PHNN is deliberately not an implemented experiment row in the current matrix.
The removed prototype treated the upstream prompt embedding as an explicit
control input `u` and applied damping to the wrong Hamiltonian derivative. It
must not be used as evidence for a PHNN result.

The current runnable comparison is:

- `hnn`: `dz/dt = J grad(H(z))`;
- `forced_damped_hnn`: `dz/dt = (J - R) grad(H(z)) + F(t)`.

Both use the upstream pathway only to construct the initial state `z0`; neither
receives a separate `u`. This matches the current data contract, which has no
independent intervention or port variable.

A future PHNN row should be added only after the dataset defines a distinct
port/control signal and its units, timing, and supervision. At that point the
implementation must state whether it follows the explicit time-dependent form

```text
qdot = dH/dp
pdot = -dH/dq + N dH/dp + F(t)
```

or a general port-Hamiltonian form with an independently observed input. A
prompt-derived initial condition is not, by itself, an external control port.

See `docs/HAMILTONIAN_EXPERIMENTS.md` for the maintained equations and
experiment IDs.
