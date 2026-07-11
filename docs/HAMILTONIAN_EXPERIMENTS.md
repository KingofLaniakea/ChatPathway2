# Hamiltonian experiment design

## Maintained vector fields

```text
HNN:                 dz/dt = J grad H(z)
forced/damped HNN:   dz/dt = (J - rI) grad H(z) + F(t)
```

`J = Q^T J0 Q`, where `Q` is learned through Householder reflections and `J0`
is the canonical full-rank Poisson matrix. Thus `J^T=-J`, `J^2=-I`, and the
model learns a symplectic frame without declaring the first 64 arbitrary AE
coordinates to be `q` and the last 64 to be `p`.

The active damping is isotropic `rI`, with `r=softplus(raw_r)>=0`. This avoids
reintroducing unsupported coordinate semantics through a diagonal latent-axis
damping matrix. `F(t)` is time-only, zero-initialized, and separately
regularized. It is not a knockout/control input `u`.

For the conservative term,

```text
grad(H)^T J grad(H) = 0.
```

For the forced/damped model,

```text
dH/dt = -r ||grad(H)||^2 + grad(H)^T F(t).
```

Consequently neither total-energy monotonicity nor convergence is guaranteed
when forcing is present. Learned `H` is a latent structural potential, not a
validated biochemical free energy or causal-direction score.

## Semantic trajectory

The dataset preserves ordered graph layers. A layer may contain several atomic
`A relation B` events at the same graph depth. Framework A locates those text
spans, pools them into one contextual layer target, and advances the ODE once
to the next graph layer. It never interprets source-list or sentence order
inside a layer as biological time.

The initial state `z0` is the frozen AE encoding of the final prompt token. The
AE is therefore trained on answer states plus that prompt anchor. For layer
targets, the fixed decoder is compared as

```text
D(z[k+1]) - D(z[k])  versus  h[k+1] - h[k],
```

not as `D(z[k+1]-z[k])`, because the decoder is nonlinear. State cosine and
latent smooth-L1 terms supplement velocity cosine. Target layer states are
stop-gradient so the auxiliary objective cannot move its own target in the
same update.

## Surrogate-time policy

One graph layer always consumes the fixed increment `dt=1/128`. At most 128
semantic layer transitions are used; longer tails are counted and ignored by
the dynamics loss, while token CE still uses the retained answer tokens. The
ODE integrates only to the largest retained layer count in the current batch.
This avoids stretching every pathway to an arbitrary common duration.

## Controlled attribution

Within each seed, all stage-2 rows share the exact SFT and AE digests:

1. stage-1 SFT direct baseline;
2. the same stage-2 loop with all dynamics losses and dynamics LR zero;
3. stage-2 SFT plus pure HNN;
4. stage-2 SFT plus forced/damped HNN.

The second row is essential: without it, an improvement could come merely from
an extra SFT stage. Three seed-scoped replicates prevent artifact overwrite and
support uncertainty estimates.

## Inference boundary

The active generation mode is direct greedy LoRA inference. HNN is a training
regularizer and Task 0 diagnostic. A token-by-token HNN rollout is invalid
because the dynamics was trained per graph layer. Rollout or mixed generation
can be reconsidered only after a tested semantic boundary controller advances
the latent exactly once per completed JSON graph layer.

## Checkpoint selection and provenance

SFT, AE, and stage-2 trainers use group-safe validation, early stopping, and a
validation-selected `checkpoint_best`. Run manifests record the Git commit,
data digest, base-model revision, shared adapter/AE digests, seed, and config.
Logs record prompt/answer truncation and semantic-layer coverage. The current
implementation saves per-epoch model artifacts but does not claim exact
optimizer/RNG resume after interruption.
