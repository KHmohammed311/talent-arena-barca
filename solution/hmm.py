"""A two-state hidden Markov model (ball dead / ball alive) fitted WITHOUT labels.

Baum-Welch on the unlabelled match. Each state emits, independently given the state:
  - a Gaussian over collective player activity,
  - Bernoulli variables (ball unseen, ball outside the pitch, ...).
The initialisation only encodes which state is which (dead = players walking); every parameter
is then re-estimated from the tracking itself.
"""

import numpy as np

DEAD, ALIVE = 0, 1


def _log_emission(x_gauss, x_bern, mu, sd, pb):
    """[T, 2] log-likelihood. x_gauss [T, G] (NaN = unobserved), x_bern [T, K] in {0, 1, NaN}."""
    T = x_gauss.shape[0]
    ll = np.zeros((T, 2))
    for s in (DEAD, ALIVE):
        z = (x_gauss - mu[s]) / sd[s]
        g = -0.5 * z * z - np.log(sd[s]) - 0.9189385332
        ll[:, s] += np.where(np.isnan(g), 0.0, g).sum(1)
        b = x_bern * np.log(pb[s]) + (1 - x_bern) * np.log(1 - pb[s])
        ll[:, s] += np.where(np.isnan(b), 0.0, b).sum(1)
    return ll


def _forward_backward(ll, logA, logpi):
    T = ll.shape[0]
    alpha = np.zeros((T, 2))
    beta = np.zeros((T, 2))
    alpha[0] = logpi + ll[0]
    for t in range(1, T):
        m = alpha[t - 1][:, None] + logA
        alpha[t] = np.logaddexp(m[0], m[1]) + ll[t]
    for t in range(T - 2, -1, -1):
        m = logA + (ll[t + 1] + beta[t + 1])[None, :]
        beta[t] = np.logaddexp(m[:, 0], m[:, 1])
    evidence = np.logaddexp(alpha[-1, 0], alpha[-1, 1])
    gamma = np.exp(alpha + beta - evidence)
    xi = np.exp(alpha[:-1, :, None] + logA[None] + (ll[1:] + beta[1:])[:, None, :] - evidence)
    return gamma, xi.sum(0), evidence


def fit(sequences, init, n_iter=30, temper=1.0, freeze_bern=()):
    """sequences: list of (x_gauss, x_bern). init: dict(mu, sd, pb, stay). Returns params.

    Bernoulli columns listed in `freeze_bern` keep their initial probabilities: used for
    "ball unseen", which would otherwise swallow the model (the states would become
    seen / unseen rather than alive / dead)."""
    pb0 = np.array(init["pb"], dtype=float)
    mu, sd, pb = [np.array(init[k], dtype=float) for k in ("mu", "sd", "pb")]
    stay = np.array(init["stay"], dtype=float)
    A = np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])
    pi = np.array([0.5, 0.5])
    history = []
    for _ in range(n_iter):
        g_acc = [np.zeros_like(mu), np.zeros_like(mu), np.zeros_like(mu)]
        b_acc = [np.zeros_like(pb), np.zeros_like(pb)]
        xi_acc = np.zeros((2, 2))
        total = 0.0
        for xg, xb in sequences:
            ll = _log_emission(xg, xb, mu, sd, pb) * temper
            gamma, xi, ev = _forward_backward(ll, np.log(A), np.log(pi))
            total += ev
            xi_acc += xi
            ok = ~np.isnan(xg)
            xg0 = np.where(ok, xg, 0.0)
            okb = ~np.isnan(xb)
            xb0 = np.where(okb, xb, 0.0)
            for s in (DEAD, ALIVE):
                w = gamma[:, s][:, None]
                g_acc[0][s] += (w * ok).sum(0)
                g_acc[1][s] += (w * xg0).sum(0)
                g_acc[2][s] += (w * xg0 * xg0).sum(0)
                b_acc[0][s] += (w * okb).sum(0)
                b_acc[1][s] += (w * xb0).sum(0)
        mu = g_acc[1] / np.maximum(g_acc[0], 1e-9)
        sd = np.sqrt(np.maximum(g_acc[2] / np.maximum(g_acc[0], 1e-9) - mu ** 2, 1e-4))
        pb = np.clip(b_acc[1] / np.maximum(b_acc[0], 1e-9), 1e-3, 1 - 1e-3)
        for k in freeze_bern:
            pb[:, k] = pb0[:, k]
        A = xi_acc / xi_acc.sum(1, keepdims=True)
        history.append(total)
    return {"mu": mu, "sd": sd, "pb": pb, "A": A, "pi": pi, "loglik": history, "temper": temper}


def posterior(xg, xb, p):
    """P(dead) per step."""
    ll = _log_emission(xg, xb, p["mu"], p["sd"], p["pb"]) * p["temper"]
    gamma, _, _ = _forward_backward(ll, np.log(p["A"]), np.log(p["pi"]))
    return gamma[:, DEAD]


def viterbi(xg, xb, p, extra=None):
    """Most likely state path. `extra` [T, 2] is added to the log-emissions (hard constraints)."""
    ll = _log_emission(xg, xb, p["mu"], p["sd"], p["pb"]) * p["temper"]
    if extra is not None:
        ll = ll + extra
    logA = np.log(p["A"])
    T = ll.shape[0]
    back = np.zeros((T, 2), dtype=np.int8)
    score = np.log(p["pi"]) + ll[0]
    for t in range(1, T):
        m = score[:, None] + logA
        back[t] = np.argmax(m, axis=0)
        score = m[back[t], [0, 1]] + ll[t]
    path = np.zeros(T, dtype=np.int8)
    s = int(np.argmax(score))
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return path
