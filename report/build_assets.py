"""
Genera tutte le figure e tutti i numeri usati nel report (report.typ).

Nessun numero del report è scritto a mano: il testo legge results.json, prodotto da
questo script a partire dal package option_pricer. Rieseguendo lo script si ottengono
esattamente gli stessi valori (tutti i semi casuali sono fissati).

Uso, dalla cartella principale del progetto:
    python report/build_assets.py
    typst compile --root . report/report.typ report/option_pricer_report.pdf

Output: report/figures/*.svg e report/results.json (numeri già formattati come testo).
"""

import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # nessuna finestra: le figure vanno solo su file

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import font_manager
from scipy.stats import norm

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent
sys.path.insert(0, str(ROOT))

from option_pricer import (  # noqa: E402
    GBM, AsianOption, BarrierOption, EuropeanCall, EuropeanPut,
    black_scholes_call, black_scholes_greeks, black_scholes_put, compare_with_black_scholes, compute_greeks,
    convergence_study, price_black_scholes, price_monte_carlo,
)
from option_pricer.plotting import BLACK, FILL, LIGHT, MID, SERIES, apply_style  # noqa: E402

FIG_DIR = REPORT_DIR / "figures"
TEXT_WIDTH = 15.2 / 2.54  # larghezza del testo in pollici: A4 (21 cm) con margini laterali di 2.9 cm

# Parametri comuni a tutto il report
S0, R, SIGMA, K, T = 100.0, 0.05, 0.20, 100.0, 1.0
B_UP, B_DOWN = 130.0, 80.0
SEED = 42


# ---------------------------------------------------------------------------
# Formattazione: numeri come stringhe pronte per il testo, con il segno meno tipografico
# ---------------------------------------------------------------------------

def num(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}".replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    if round(x, digits) == 0:
        return f"{0:.{digits}f}"  # niente "−0.00" o "+0.00"
    return f"{x:+.{digits}f}".replace("-", "−")


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def integer(n: int) -> str:
    # Spazio sottile tra le migliaia (convenzione SI): 200000 -> "200 000"
    return f"{n:,}".replace(",", " ")


def sci(x: float) -> str:
    mantissa, exponent = f"{abs(x):.1e}".split("e")
    return f"{mantissa} × 10^{int(exponent)}"


# ---------------------------------------------------------------------------
# Stile delle figure: quello accademico dei notebook, con il font del report
# ---------------------------------------------------------------------------

def setup_style() -> None:
    apply_style()
    for font_file in (REPORT_DIR / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font_file))
    plt.rcParams.update({
        # Libertinus Serif, come il testo del report; formule nello stesso font
        "font.serif": ["Libertinus Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Libertinus Serif",
        "mathtext.it": "Libertinus Serif:italic",
        "mathtext.bf": "Libertinus Serif:bold",
        # Le figure entrano nel report alla loro dimensione reale: font di poco più piccoli del testo (11 pt)
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        # SVG con il testo convertito in tracciati: il report non dipende dai font installati
        "svg.fonttype": "path",
    })


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Formule chiuse di riferimento per le esotiche (usate solo come verifica indipendente)
# ---------------------------------------------------------------------------

def geometric_asian_call_discrete(S0, K, T, r, sigma, n):
    """
    Call su media GEOMETRICA di n osservazioni equispaziate t_i = iT/n: formula esatta,
    perché il logaritmo della media geometrica è normale (Kemna e Vorst, 1990).
    """
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d = (mu - np.log(K)) / np.sqrt(var)
    return np.exp(-r * T) * (np.exp(mu + 0.5 * var) * norm.cdf(d + np.sqrt(var)) - K * norm.cdf(d))


def up_and_out_call_continuous(S0, K, B, T, r, sigma):
    """Call up-and-out con barriera osservata in continuo, B > K (Reiner e Rubinstein, 1991)."""
    vol = sigma * np.sqrt(T)
    lam = (r + 0.5 * sigma**2) / sigma**2
    disc = np.exp(-r * T)
    x1 = np.log(S0 / B) / vol + lam * vol
    y = np.log(B**2 / (S0 * K)) / vol + lam * vol
    y1 = np.log(B / S0) / vol + lam * vol
    up_and_in = (S0 * norm.cdf(x1) - K * disc * norm.cdf(x1 - vol)
                 - S0 * (B / S0) ** (2 * lam) * (norm.cdf(-y) - norm.cdf(-y1))
                 + K * disc * (B / S0) ** (2 * lam - 2) * (norm.cdf(-y + vol) - norm.cdf(-y1 + vol)))
    return black_scholes_call(S0, K, T, r, sigma) - up_and_in


def up_and_out_call_bgk(S0, K, B, T, r, sigma, n):
    """
    Approssimazione per osservazione DISCRETA su n date: formula continua con la barriera
    spostata a B·exp(β σ √(T/n)), β = −ζ(1/2)/√(2π) ≈ 0.5826 (Broadie, Glasserman e Kou, 1997).
    """
    beta = 0.5826
    return up_and_out_call_continuous(S0, K, B * np.exp(beta * sigma * np.sqrt(T / n)), T, r, sigma)


# ---------------------------------------------------------------------------
# Sezione 3: validazione contro Black-Scholes
# ---------------------------------------------------------------------------

def validation(results: dict) -> None:
    model = GBM(S0=S0, r=R, sigma=SIGMA)
    n_paths = 200_000

    # Put-call parity su input casuali: verifica le formule di Black-Scholes senza Monte Carlo
    rng = np.random.default_rng(0)
    worst_gap = 0.0
    for _ in range(10_000):
        s0, k = rng.uniform(1, 300, size=2)
        t, rate, vol = rng.uniform(0.01, 10), rng.uniform(-0.02, 0.15), rng.uniform(0.01, 1.5)
        gap = black_scholes_call(s0, k, t, rate, vol) - black_scholes_put(s0, k, t, rate, vol) - (s0 - k * np.exp(-rate * t))
        worst_gap = max(worst_gap, abs(gap))
    mantissa, exponent = f"{worst_gap:.1e}".split("e")
    results["parity"] = {"cases": integer(10_000), "mantissa": mantissa,
                         "exponent": str(int(exponent)).replace("-", "−")}

    base = {}
    for name, instrument in [("call", EuropeanCall(K=K, T=T)), ("put", EuropeanPut(K=K, T=T))]:
        res = compare_with_black_scholes(model, instrument, n_paths=n_paths, seed=SEED)
        base[name] = {"mc": num(res.mc.price), "se": num(res.mc.std_error), "bs": num(res.bs_price),
                      "z": signed(res.z_score), "ci_low": num(res.mc.ci_low), "ci_high": num(res.mc.ci_high)}
    results["base"] = {"N": integer(n_paths), **base}

    # Scenari: stessi del notebook 02 (seme diverso per scenario, call e put sulle stesse traiettorie)
    scenarios = [
        ("At the money", 100.0, 100.0, 1.0, 0.20),
        ("In/out of the money", 100.0, 80.0, 1.0, 0.20),
        ("Out of/in the money", 100.0, 120.0, 1.0, 0.20),
        ("High volatility", 100.0, 100.0, 1.0, 0.50),
        ("Short maturity", 100.0, 100.0, 1 / 12, 0.20),
        ("Long maturity", 100.0, 100.0, 5.0, 0.20),
    ]
    rows, z_values, outside = [], [], 0
    for i, (label, s0, k, t, vol) in enumerate(scenarios):
        scenario_model = GBM(S0=s0, r=R, sigma=vol)
        for kind, Option in [("call", EuropeanCall), ("put", EuropeanPut)]:
            res = compare_with_black_scholes(scenario_model, Option(K=k, T=t), n_paths=n_paths, seed=SEED + i)
            z_values.append(res.z_score)
            outside += not res.within_ci
            rows.append({"label": label, "K": f"{k:g}", "T": "1/12" if t == 1 / 12 else f"{t:g}",
                         "sigma": f"{vol:.2f}", "type": kind, "bs": num(res.bs_price),
                         "mc": num(res.mc.price), "se": num(res.mc.std_error), "z": signed(res.z_score)})
    results["scenarios"] = {"rows": rows, "max_abs_z": num(max(abs(z) for z in z_values), 2),
                            "outside": outside, "count": len(rows)}

    # Il caso fuori dall'intervallo ripetuto con semi indipendenti: z deve essere ~ N(0, 1)
    runs, run_paths = 500, 20_000
    itm_model, itm_call = GBM(S0=100.0, r=R, sigma=0.20), EuropeanCall(K=80.0, T=1.0)
    z = np.array([compare_with_black_scholes(itm_model, itm_call, n_paths=run_paths, seed=1_000 + j).z_score
                  for j in range(runs)])
    results["repeat"] = {"runs": integer(runs), "N": integer(run_paths), "mean_z": signed(z.mean(), 3),
                         "mean_z_se": num(1 / np.sqrt(runs), 3), "sd_z": num(z.std(ddof=1), 3),
                         "coverage": pct(np.mean(np.abs(z) < norm.ppf(0.975)))}

    # Figura 1: convergenza verso Black-Scholes, stime indipendenti per ciascun N
    n_values = np.unique(np.geomspace(100, 1_000_000, 25).astype(int))
    z95 = norm.ppf(0.975)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.55))
    inside_counts = {}
    for ax, panel, (name, instrument) in zip(axes, "ab", [("call", EuropeanCall(K=K, T=T)),
                                                          ("put", EuropeanPut(K=K, T=T))]):
        estimates = convergence_study(model, instrument, n_values, seed=SEED)
        prices = np.array([e.price for e in estimates])
        bs_price = price_black_scholes(model, instrument)
        s_payoff = estimates[-1].std_error * np.sqrt(n_values[-1])  # da SE = s/√N
        n_grid = np.geomspace(n_values[0], n_values[-1], 200)
        band = z95 * s_payoff / np.sqrt(n_grid)
        inside_counts[name] = int(np.sum(np.abs(prices - bs_price) <= z95 * s_payoff / np.sqrt(n_values)))

        ax.fill_between(n_grid, bs_price - band, bs_price + band, color=FILL, linewidth=0, label="Expected 95% band")
        ax.axhline(bs_price, color=BLACK, linewidth=0.9, label="Black–Scholes")
        ax.plot(n_values, prices, "o", color=BLACK, markerfacecolor="white", markeredgewidth=0.8,
                label="Monte Carlo estimate")
        ax.set_xscale("log")
        ax.set_title(f"({panel}) European {name}")
        ax.set_xlabel(r"Number of paths $N$")
        ax.set_ylabel("Price")
        ax.legend(loc="upper right")
    fig.tight_layout(w_pad=2)
    save(fig, "convergence")
    results["convergence"] = {"points": len(n_values), **inside_counts}


# ---------------------------------------------------------------------------
# Sezione 4: opzioni path-dependent
# ---------------------------------------------------------------------------

def exotics(results: dict) -> None:
    model = GBM(S0=S0, r=R, sigma=SIGMA)
    n_paths, n_steps = 100_000, 252

    book = [
        ("European call", EuropeanCall(K=K, T=T)),
        ("Asian call", AsianOption(K=K, T=T, kind="call")),
        ("Up-and-out call", BarrierOption(K=K, T=T, barrier=B_UP, kind="call", direction="up", knock="out")),
        ("Up-and-in call", BarrierOption(K=K, T=T, barrier=B_UP, kind="call", direction="up", knock="in")),
        ("European put", EuropeanPut(K=K, T=T)),
        ("Asian put", AsianOption(K=K, T=T, kind="put")),
        ("Down-and-out put", BarrierOption(K=K, T=T, barrier=B_DOWN, kind="put", direction="down", knock="out")),
        ("Down-and-in put", BarrierOption(K=K, T=T, barrier=B_DOWN, kind="put", direction="down", knock="in")),
    ]
    prices = {name: price_monte_carlo(model, contract, n_paths=n_paths, n_steps=n_steps, seed=SEED)
              for name, contract in book}
    call_gap = prices["Up-and-out call"].price + prices["Up-and-in call"].price - prices["European call"].price
    put_gap = prices["Down-and-out put"].price + prices["Down-and-in put"].price - prices["European put"].price
    eu_check = compare_with_black_scholes(model, EuropeanCall(K=K, T=T), n_paths=n_paths, n_steps=n_steps, seed=SEED)
    results["exotics"] = {
        "N": integer(n_paths), "steps": n_steps,
        "rows": [{"name": name, "mc": num(p.price), "se": num(p.std_error)} for name, p in prices.items()],
        "parity_gap": max(abs(call_gap), abs(put_gap)),
        "european_252_z": signed(eu_check.z_score),
        "asian_ratio": pct(prices["Asian call"].price / prices["European call"].price, 0),
    }

    # Figura 2: prezzo al variare del numero di date di osservazione
    steps_grid = [2, 4, 12, 52, 252]
    contracts = [("European call", book[0][1]), ("Asian call", book[1][1]), ("Up-and-out call", book[2][1])]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.72, 2.6))
    monitoring = {}
    for (label, contract), style in zip(contracts, SERIES):
        estimates = [price_monte_carlo(model, contract, n_paths=n_paths, n_steps=n, seed=SEED) for n in steps_grid]
        values = np.array([e.price for e in estimates])
        half_widths = np.array([e.ci_high - e.price for e in estimates])
        monitoring[label] = [num(v, 2) for v in values]
        ax.errorbar(steps_grid, values, yerr=half_widths, capsize=2, elinewidth=0.7, markerfacecolor="white",
                    markeredgewidth=0.8, label=label, **style)
    ax.set_xscale("log")
    ax.set_xticks(steps_grid, [str(n) for n in steps_grid])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"Number of monitoring dates $n$ per year")
    ax.set_ylabel("Price")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.64))  # nello spazio libero tra europea e asiatica
    fig.tight_layout()
    save(fig, "monitoring")
    results["monitoring"] = {"steps": steps_grid, **monitoring}

    # Figura 3: distribuzione di S_T e della media A
    paths = model.simulate_paths(T=T, n_steps=n_steps, n_paths=n_paths, rng=np.random.default_rng(SEED))
    S_T, A = paths[:, -1], paths[:, 1:].mean(axis=1)
    bins = np.linspace(np.quantile(S_T, 0.002), np.quantile(S_T, 0.998), 91)
    weights = np.full(n_paths, 1 / (n_paths * (bins[1] - bins[0])))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.72, 2.5))
    ax.hist(A, bins=bins, weights=weights, histtype="stepfilled", color=LIGHT, label=r"Arithmetic average $A$")
    ax.hist(S_T, bins=bins, weights=weights, histtype="step", color=BLACK, linewidth=0.9, label=r"Terminal price $S_T$")
    ax.axvline(K, color=BLACK, linewidth=0.8, linestyle=":", label=rf"Strike $K={K:g}$")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Price")
    ax.set_ylabel("Probability density")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "asian_distribution")
    results["asian"] = {"sd_ST": num(S_T.std(ddof=1), 1), "sd_A": num(A.std(ddof=1), 1),
                        "logvol_ST": pct(np.log(S_T).std(ddof=1)), "logvol_A": pct(np.log(A).std(ddof=1)),
                        "sigma_sqrt3": pct(SIGMA / np.sqrt(3)),
                        "itm_ST": pct(np.mean(S_T > K)), "itm_A": pct(np.mean(A > K))}

    # Figura 4: dipendenza dal percorso della up-and-out
    up_out = book[2][1]
    hit, payoff = up_out.touched(paths), up_out.payoff(paths)
    between = (S_T > K) & (S_T < B_UP)
    ko = np.flatnonzero(hit)[np.argmin(np.abs(S_T[hit] - 115.0))]
    alive = np.flatnonzero(~hit)[np.argmin(np.abs(S_T[~hit] - S_T[ko]))]
    first_touch = int(np.argmax(paths[ko] >= B_UP))
    t = np.linspace(0, T, n_steps + 1)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7))
    ax_a.plot(t, paths[alive], color=BLACK, linewidth=0.8, label="Barrier never hit")
    ax_a.plot(t, paths[ko], color=MID, linewidth=0.8, label="Barrier hit")
    ax_a.plot(t[first_touch], paths[ko, first_touch], "o", color=BLACK, markerfacecolor="white",
              markeredgewidth=0.9, markersize=5, zorder=3, label="Knock-out")
    ax_a.axhline(B_UP, color=BLACK, linewidth=0.8, linestyle="--", label=rf"Barrier $B={B_UP:g}$")
    ax_a.axhline(K, color=BLACK, linewidth=0.8, linestyle=":", label=rf"Strike $K={K:g}$")
    y_low = min(paths[[alive, ko]].min(), K) - 5
    ax_a.set_ylim(y_low, B_UP + 1.05 * (B_UP - y_low))  # spazio in alto per la legenda
    ax_a.set_xlim(0, T)
    ax_a.set_title("(a) Two paths with the same terminal price")
    ax_a.set_xlabel(r"Time $t$ (years)")
    ax_a.set_ylabel(r"Underlying price $S_t$")
    ax_a.legend(loc="upper left", ncol=2)

    sample = slice(0, 5_000)
    S_s, pay_s, hit_s = S_T[sample], payoff[sample], hit[sample]
    x = np.linspace(S_s.min(), S_s.max(), 400)
    ax_b.axvspan(K, B_UP, color=FILL, linewidth=0, zorder=0, label=r"$K<S_T<B$")
    ax_b.plot(x, np.maximum(x - K, 0.0), color=BLACK, linewidth=0.8, linestyle=":", zorder=1, label="European call payoff")
    ax_b.scatter(S_s[~hit_s], pay_s[~hit_s], s=3, color=BLACK, linewidths=0, zorder=2, rasterized=True,
                 label="Barrier never hit")
    ax_b.scatter(S_s[hit_s], pay_s[hit_s], s=3, color=MID, linewidths=0, zorder=2, rasterized=True,
                 label="Barrier hit")
    ax_b.axvline(B_UP, color=BLACK, linewidth=0.8, linestyle="--")
    ax_b.set_title(r"(b) Payoff against $S_T$")
    ax_b.set_xlabel(r"Terminal price $S_T$")
    ax_b.set_ylabel("Payoff")
    ax_b.legend(loc="upper left", markerscale=3)
    fig.tight_layout(w_pad=2)
    save(fig, "barrier")
    results["barrier"] = {"S_T": num(S_T[ko], 1), "alive_payoff": num(payoff[alive], 2),
                          "touch_time": num(t[first_touch], 2), "hit_share": pct(hit.mean()),
                          "hit_share_between": pct(hit[between].mean()), "sample": integer(5_000)}

    # Verifiche indipendenti con formule chiuse: 20 simulazioni indipendenti, stime aggregate
    runs = 20
    sums = {key: [0.0, 0.0] for key in ("geometric", "arithmetic", "up_out")}
    asian_call, disc = book[1][1], np.exp(-R * T)
    for run in range(runs):
        run_paths = model.simulate_paths(T=T, n_steps=n_steps, n_paths=n_paths, rng=np.random.default_rng(2_000 + run))
        geometric = np.exp(np.log(run_paths[:, 1:]).mean(axis=1))
        samples = {"geometric": disc * np.maximum(geometric - K, 0.0),
                   "arithmetic": disc * asian_call.payoff(run_paths),
                   "up_out": disc * up_out.payoff(run_paths)}
        for key, x_run in samples.items():
            sums[key][0] += x_run.sum()
            sums[key][1] += (x_run**2).sum()
    total = runs * n_paths

    def pooled(key):
        mean = sums[key][0] / total
        se = np.sqrt((sums[key][1] / total - mean**2) * total / (total - 1) / total)
        return mean, se

    geo_mc, geo_se = pooled("geometric")
    ari_mc, ari_se = pooled("arithmetic")
    uo_mc, uo_se = pooled("up_out")
    geo_ref = geometric_asian_call_discrete(S0, K, T, R, SIGMA, n_steps)
    uo_ref = up_and_out_call_bgk(S0, K, B_UP, T, R, SIGMA, n_steps)
    results["reference_checks"] = {
        "N": integer(total), "runs": runs,
        "geometric": {"ref": num(geo_ref), "mc": num(geo_mc), "se": num(geo_se), "z": signed((geo_mc - geo_ref) / geo_se)},
        "arithmetic": {"mc": num(ari_mc), "se": num(ari_se)},
        "up_out": {"ref": num(uo_ref), "mc": num(uo_mc), "se": num(uo_se), "z": signed((uo_mc - uo_ref) / uo_se),
                   "continuous": num(up_and_out_call_continuous(S0, K, B_UP, T, R, SIGMA))},
    }


# ---------------------------------------------------------------------------
# Sezione 5: greeks
# ---------------------------------------------------------------------------

def greeks(results: dict) -> None:
    model = GBM(S0=S0, r=R, sigma=SIGMA)
    n_paths = 100_000
    call = EuropeanCall(K=K, T=T)

    exact = black_scholes_greeks(model, call)
    crn = compute_greeks(model, call, seed=SEED, n_paths=n_paths)
    independent = compute_greeks(model, call, seed=SEED, n_paths=n_paths, common_random_numbers=False)
    european, z_values = [], []
    for name in ("delta", "gamma", "vega"):
        est, ind = getattr(crn, name), getattr(independent, name)
        ratio = ind.std_error / est.std_error
        z_values.append((est.value - exact[name]) / est.std_error)
        european.append({"name": name.capitalize(), "mc": num(est.value), "se": num(est.std_error),
                         "bs": num(exact[name]), "z": signed(z_values[-1]),
                         "se_indep": num(ind.std_error), "ratio": f"{ratio:.0f}", "paths_factor": integer(round(ratio**2))})
    results["greeks_european"] = {"N": integer(n_paths), "rows": european,
                                  "max_abs_z": num(max(abs(z) for z in z_values), 2)}

    # Greeks delle esotiche (osservazione giornaliera, perturbazione dell'1%)
    n_steps = 252
    up_out = BarrierOption(K=K, T=T, barrier=B_UP, kind="call", direction="up", knock="out")
    exotic_rows = []
    for name, contract in [("European call", call), ("Asian call", AsianOption(K=K, T=T, kind="call")),
                           ("Up-and-out call", up_out)]:
        g = compute_greeks(model, contract, seed=SEED, n_paths=n_paths, n_steps=n_steps)
        exotic_rows.append({"name": name, **{key: {"v": num(getattr(g, key).value), "se": num(getattr(g, key).std_error)}
                                             for key in ("delta", "gamma", "vega")}})

    # Riferimento per la up-and-out: differenze finite sulla formula con correzione BGK (deterministica)
    def uo(s0, sigma):
        return up_and_out_call_bgk(s0, K, B_UP, T, R, sigma, n_steps)

    h, k = 1e-2, 1e-4
    reference = {"delta": num((uo(S0 + h, SIGMA) - uo(S0 - h, SIGMA)) / (2 * h)),
                 "gamma": num((uo(S0 + h, SIGMA) - 2 * uo(S0, SIGMA) + uo(S0 - h, SIGMA)) / h**2),
                 "vega": num((uo(S0, SIGMA + k) - uo(S0, SIGMA - k)) / (2 * k))}

    bumps = []
    for bump in (0.01, 0.02, 0.05):
        g = compute_greeks(model, up_out, seed=SEED, n_paths=n_paths, n_steps=n_steps, spot_bump=bump)
        bumps.append({"bump": f"{bump:.0%}", "delta": num(g.delta.value), "delta_se": num(g.delta.std_error),
                      "gamma": num(g.gamma.value), "gamma_se": num(g.gamma.std_error)})
    results["greeks_exotic"] = {"rows": exotic_rows, "up_out_reference": reference, "bumps": bumps}


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    setup_style()
    results = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
               "parameters": {"S0": f"{S0:g}", "r": f"{R:g}", "sigma": f"{SIGMA:.2f}", "K": f"{K:g}",
                              "T": f"{T:g}", "B_up": f"{B_UP:g}", "B_down": f"{B_DOWN:g}", "seed": SEED}}
    validation(results)
    exotics(results)
    greeks(results)
    results["exotics"]["parity_gap"] = "0" if results["exotics"]["parity_gap"] == 0 else sci(results["exotics"]["parity_gap"])
    (REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
