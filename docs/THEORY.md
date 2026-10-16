# Theory: error budgets and the selection criterion

This note derives the numerical bounds behind the AKO correctness gate and the
statistical rule behind "kernel A is faster than kernel B". Each section ends
with what the code actually does and where it departs from the derivation.
Bounds marked *tested* are checked numerically on CPU by
`tests/test_numerical_bounds.py` (fp64 reference, repository implementations in
fp16 / bf16 / fp32).

Notation. $u$ is the unit roundoff of a format: $u_{32} = 2^{-24}$ (fp32),
$u_{16} = 2^{-11}$ (fp16), $u_{bf} = 2^{-8}$ (bf16). One ulp of a value $y$ is at
most $2u|y|$. The standard model is $\mathrm{fl}(a \circ b) = (a \circ b)(1+\delta)$,
$|\delta| \le u$, for $\circ \in \{+,-,\times,/\}$.

## 1. Summation

**Deterministic.** If $|\delta_i| \le u$, $\rho_i = \pm 1$ and $nu < 1$, then
$\prod_{i=1}^{n}(1+\delta_i)^{\rho_i} = 1 + \theta_n$ with

$$|\theta_n| \le \gamma_n := \frac{nu}{1-nu}.$$

This is Lemma 3.1 of Higham [1] (cited with that number, and restated as
Lemma 2.2, in [2]). In any summation order, the term $x_i$ passes through at most
$h$ additions, where $h$ is the depth of the summation tree, so

$$\Bigl|\mathrm{fl}\Bigl(\sum_i x_i\Bigr) - \sum_i x_i\Bigr| \le \gamma_h \sum_i |x_i| .$$

Recursive (left-to-right) summation has $h = n-1$. A balanced binary tree
(pairwise summation, and the shape of a GPU block reduction) has
$h = \lceil \log_2 n \rceil$. See [1, ch. 4] for the general summation algorithm
and its error analysis. *Tested* for `torch.sum` in fp32 ($\gamma_{n-1}$, valid for
any order) and for an explicit pairwise sum ($\gamma_{\lceil \log_2 n\rceil}$).

**Probabilistic.** If the $\delta_i$ are independent, of mean zero and bounded by
$u$, then for any $\lambda > 0$, $|\theta_n| \le \tilde\gamma_n(\lambda)$ with
probability at least $P(\lambda) = 1 - 2\exp\bigl(-\lambda^2(1-u)^2/2\bigr)$, where

$$\tilde\gamma_n(\lambda) = \exp\Bigl(\lambda\sqrt{n}\,u + \frac{nu^2}{1-u}\Bigr) - 1
= \lambda\sqrt{n}\,u + O(u^2)$$

(Higham & Mary [2], Theorem 2.4). For an inner product of length $n$ every term
satisfies the bound simultaneously with probability at least
$Q(\lambda, n) = 1 - n\,(1 - P(\lambda))$ ([2], Theorem 3.1). With $\lambda = 6$ and
$n = 4096$, $Q \approx 0.9999$. Independence is a model, not a guarantee: the
deterministic bound is the one a test can rely on. On seeded data the
probabilistic bound is *tested* too.

## 2. RMSNorm: a derived budget against the coded one

The reference (`ako/ops/rmsnorm_ref.py`) upcasts fp16/bf16 inputs to fp32,
computes $y_i = w_i x_i / \sqrt{m + \varepsilon}$ with $m = \frac1n\sum_j x_j^2$,
and rounds once to the input dtype. Inputs and weights are exact in fp32.

Error of the fp32 part, counting roundings:

- $n$ squares and $n-1$ additions, then $/n$ and $+\varepsilon$: every term sees
  at most $n+2$ roundings and all terms are positive, so
  $\widehat{m+\varepsilon} = (m+\varepsilon)(1+\theta_{n+2})$.
- `rsqrt`: assumed accurate to $2u_{32}$ (it is not a correctly rounded
  operation). $(1+\theta)^{-1/2}$ halves the relative error to first order.
- two multiplications ($x \cdot r$, then $\cdot\, w$).

$$\delta_{32}(n) = (1-\gamma_{n+2})^{-1/2}(1+u_{32})^{6} - 1
\;\approx\; \tfrac12 (n+2)\,u_{32} + 4u_{32} .$$

After the final rounding to the output format with roundoff $u$,

$$|\hat y_i - y_i| \le \bigl(u + (1+u)\,\delta_{32}(n)\bigr)\,|y_i| . \qquad\text{(tested)}$$

Replacing $\gamma_{n+2}$ by the probabilistic $\tilde\gamma_{n+2}(\lambda)$ gives
$\delta_{32} \approx \frac12\lambda\sqrt{n+2}\,u_{32} + 4u_{32}$.

Two correct implementations may sit on either side of $y$, so the budget for
*candidate vs reference* is $B(n) = 2\,(u + \delta_{32}(n))\,\max_i|y_i|$.

The gate (`ako/validation/numerical.py`, `DTYPE_RTOL` and `error_budget`) uses
$\mathrm{atol} + \mathrm{rtol}\cdot\max|\mathrm{ref}|$ with $\mathrm{atol} = 10^{-5}$ and
`rtol` copied from the `torch.testing.assert_close` defaults, not derived.
Comparing $\mathrm{rtol}$ with $B(n)/\max|y|$ ($\lambda = 6$):

| dtype | rtol (coded) | rtol / u | $B/\max y$, $n=4096$, deterministic | same, probabilistic |
|---|---:|---:|---:|---:|
| fp16 | 1.0e-3 | 2.05 | 1.22e-3 | 1.00e-3 |
| bf16 | 1.6e-2 | 4.10 | 8.06e-3 | 7.84e-3 |
| fp32 | 1.3e-6 | 21.8 | 2.45e-4 | 2.35e-5 |

Honest reading:

- **bf16**: the coded budget is about twice the worst-case bound. A correct
  fp32-accumulating kernel cannot be rejected; one that is off by up to ~2 ulp
  at the largest output passes too.
- **fp16**: the coded budget is one ulp at the largest magnitude plus 2.4 %. It
  is below the deterministic bound for $n \ge 4096$ and equal to the
  probabilistic one. A correct kernel passes in practice because the fp32
  reduction error is far below half an fp16 ulp, so two outputs differ by
  rounding only. A kernel that accumulates in fp16 is rejected, by design.
- **fp32**: the coded budget is 18x tighter than the probabilistic bound at
  $n = 4096$ and 190x tighter than the worst case. This analysis does not
  justify it. It holds empirically because two fp32 reductions of the same data
  usually differ by far less than the bound allows. A correct fp32 kernel with a
  different reduction order could be rejected on long rows; the gate would then
  fall back to the baseline (safe, but slower). Deriving `rtol` from $B(n)$ is
  the fix if this is ever observed.

## 3. RoPE

For position $m$ and pair $i$, RoPE applies the rotation
$R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\ \sin\theta & \cos\theta\end{pmatrix}$
with $\theta = m\,\omega_i$, $\omega_i = \mathrm{base}^{-2i/d}$
(RoFormer [3], §3.2.2, eq. (15); §3.3 uses this 0-based indexing).
$R(\theta)$ is orthogonal ([3], §3.2.2), so $\|R(\theta)x\| = \|x\|$ for every
pair. *Tested* in fp64.

**Angle error.** For two angles,
$\|R(\hat\theta) - R(\theta)\|_2 = 2|\sin(\delta\theta/2)| \le |\delta\theta|$, so

$$\|(R(\hat\theta) - R(\theta))\,x\| \le |\delta\theta|\,\|x\| .$$

`build_rope_tables` computes $\omega_i$ in fp32 through a power, then
$\hat\theta = \mathrm{fl}(m\,\hat\omega_i)$. The exponent $2i/d$ carries one rounding,
amplified by $\ln(\mathrm{base})$ through the power; the power, the reciprocal and
the product add a few more. To first order
$|\delta\theta| \le c\,u_{32}\,|\theta|$ with $c = 2\ln(\mathrm{base}) + 4 \approx 22$
for base $10^4$. The table entries are then rounded ($\cos$ and $\sin$ are
accurate to about one ulp), which adds at most $3u_{32}\|x\|$:

$$\|\Delta y\| \le \bigl(c\,u_{32}\,m\,\omega_i + 3u_{32}\bigr)\,\|x\| . \qquad\text{(tested)}$$

**Consequence.** The error is proportional to the position. At $m = 2^{16}$ and
$\omega_0 = 1$ the worst case is $22 \cdot 2^{-24} \cdot 2^{16} \approx 0.09$ rad,
far above an fp16 ulp at magnitude 1. fp32 tables are fine for the short
contexts measured in this repository. Long contexts want the angle formed in
fp64 (what `build_rope_tables(..., dtype=torch.float64)` does) and cast
afterwards.

## 4. SwiGLU: first-order propagation

$y = \mathrm{silu}(g)\,v$ with $\mathrm{silu}(g) = g\,\sigma(g)$ and
$\mathrm{silu}'(g) = \sigma(g)\bigl(1 + g(1-\sigma(g))\bigr)$.
Perturb the inputs relatively, $\hat g = g(1+\epsilon_g)$, $\hat v = v(1+\epsilon_v)$:

$$\frac{|\Delta y|}{|y|} \le \kappa(g)\,|\epsilon_g| + |\epsilon_v| + O(\epsilon^2),
\qquad \kappa(g) = \bigl|1 + g\,(1-\sigma(g))\bigr| \le 1 + |g| .$$

$\kappa(g) \to 1$ for $g \gg 0$, peaks near 1.28 at $g \approx 1.28$, and grows
like $|g|$ only for $g \ll 0$, where $y$ itself is exponentially small. Inputs
produced by a low-precision matmul carry $|\epsilon| \le u$, so the activation
turns them into at most $(\kappa(g) + 1)\,u$ relative error before its own fp32
evaluation and final rounding. *Tested* by rounding fp64 inputs to fp16/bf16 and
evaluating in fp64.

## 5. Selection criterion

Latencies are noisy and heavy-tailed, so AKO compares medians, not means.
`compare_paired` (`ako/benchmark/bench_op.py`) times A and B in alternation for
$k$ pairs (order flipped every pair, so drift hits both sides), forms the
per-pair ratios $r_j = t_A^{(j)} / t_B^{(j)}$, and reports a seeded percentile
bootstrap 95 % interval of their median.

**Rule.** B is declared faster than A only if the interval lies entirely above 1,
slower only if it lies entirely below 1. An interval containing 1 means "no
detectable difference", whatever the point estimate. This is the `significant`
field of every paired report.

For unpaired samples (two independent sweeps) the same rule applies to the
ratio of medians $\mathrm{med}(t_A)/\mathrm{med}(t_B)$, with A and B resampled
independently.

Not done: the per-shape selector (`KernelSelector`) still keeps the lowest
median without an interval, so on a near-tie it may cache the noisier winner.
That is safe for correctness (every candidate passed the gate), but it is not a
performance claim.

## References

1. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
   SIAM, 2002. https://doi.org/10.1137/1.9780898718027. Ch. 3 (Lemma 3.1,
   $\gamma_n$) and ch. 4 (summation). The lemma number was checked through its
   citation in [2]; the book itself was not consulted for this note.
2. N. J. Higham and T. Mary, "A New Approach to Probabilistic Rounding Error
   Analysis", *SIAM J. Sci. Comput.* 41(5), A2815-A2835, 2019.
   https://doi.org/10.1137/18M1226312. Preprint:
   https://eprints.maths.manchester.ac.uk/2731/ (Theorems 2.4 and 3.1 checked).
3. J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, Y. Liu, "RoFormer: Enhanced
   Transformer with Rotary Position Embedding", arXiv:2104.09864.
   https://arxiv.org/abs/2104.09864 (§3.2.2, eq. (15), §3.3 checked).
