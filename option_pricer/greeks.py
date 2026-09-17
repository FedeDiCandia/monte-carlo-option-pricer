"""
Greeks Monte Carlo con differenze finite e Common Random Numbers (CRN).

Un greek è la sensibilità del prezzo a un parametro. Con Monte Carlo non abbiamo
una formula da derivare, quindi approssimiamo la derivata "perturbando" il parametro
e riprezzando:

    Delta ≈ [V(S0 + h) - V(S0 - h)] / (2h)
    Gamma ≈ [V(S0 + h) - 2 V(S0) + V(S0 - h)] / h²
    Vega  ≈ [V(σ + k) - V(σ - k)] / (2k)

(differenze CENTRALI: errore di approssimazione O(h²) invece di O(h) di quelle in avanti).

Il problema: V(S0 + h) e V(S0 - h) sono due stime rumorose di numeri vicinissimi,
e il rumore divide per 2h (o per h²!) esplode. La soluzione è simulare i prezzi
perturbati con gli STESSI shock casuali Z (stesso seed): i due errori statistici
sono quasi identici e nella differenza si cancellano, perché
Var(X - Y) = Var(X) + Var(Y) - 2 Cov(X, Y) e con gli stessi Z la covarianza è altissima.

Come il motore, questo modulo non sa quale contratto sta trattando: funziona per
europee ed esotiche, perché usa solo discounted_payoffs(model, instrument, ...).
"""

from dataclasses import dataclass, replace

import numpy as np

from .monte_carlo import discounted_payoffs


@dataclass(frozen=True)
class Estimate:
    """Una stima Monte Carlo generica: valore e suo errore standard."""

    value: float
    std_error: float

    def __str__(self) -> str:
        return f"{self.value:.4f} ± {self.std_error:.4f}"


@dataclass(frozen=True)
class GreeksResult:
    """Prezzo e greeks di un contratto, ciascuno con il proprio errore standard."""

    price: Estimate
    delta: Estimate   # ∂V/∂S0: variazione del prezzo per +1 sul sottostante
    gamma: Estimate   # ∂²V/∂S0²: variazione del delta per +1 sul sottostante
    vega: Estimate    # ∂V/∂sigma: variazione per +1.00 di volatilità (per 1 punto di vol: vega / 100)
    common_random_numbers: bool

    def __str__(self) -> str:
        return (
            f"Prezzo {self.price}  |  Delta {self.delta}  |  Gamma {self.gamma}  |  "
            f"Vega {self.vega}  |  CRN: {'sì' if self.common_random_numbers else 'no'}"
        )


def _estimate(samples: np.ndarray) -> Estimate:
    # Media campionaria e il suo errore standard s / sqrt(N), come in price_monte_carlo
    return Estimate(
        value=float(samples.mean()),
        std_error=float(samples.std(ddof=1) / np.sqrt(samples.size)),
    )


def compute_greeks(
    model,
    instrument,
    *,
    seed: int,
    n_paths: int = 100_000,
    n_steps: int = 1,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
    common_random_numbers: bool = True,
) -> GreeksResult:
    """
    Prezzo, Delta, Gamma e Vega di `instrument` sotto `model` con differenze finite centrali.

    Parametri
    ---------
    seed      : obbligatorio. Le CRN funzionano solo se tutte le simulazioni ripartono
                dallo stesso seme: con seed=None ognuna userebbe numeri diversi, in silenzio
    n_steps   : 1 per le europee; > 1 per le esotiche (date di osservazione del contratto)
    spot_bump : perturbazione RELATIVA dello spot, h = spot_bump * S0 (0.01 = 1%)
    vol_bump  : perturbazione ASSOLUTA della volatilità, k = vol_bump (0.01 = 1 punto di vol)
    common_random_numbers : False usa un seme diverso per ogni simulazione, solo per
                            mostrare quanto rumore le CRN eliminano

    Scelta di h: troppo grande -> errore di approssimazione della derivata (bias);
    troppo piccolo -> la differenza è dominata dal rumore (varianza). Per payoff
    discontinui (le barriere) il rumore cresce molto al ridursi di h.
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("compute_greeks richiede un seed intero: senza, le CRN non sono possibili")

    h = spot_bump * model.S0
    k = vol_bump

    # Le 5 versioni del modello da prezzare. `replace` crea una copia del dataclass
    # (frozen) cambiando un solo campo: l'originale resta intatto
    scenarios = {
        "base": model,
        "spot_up": replace(model, S0=model.S0 + h),
        "spot_down": replace(model, S0=model.S0 - h),
        "vol_up": replace(model, sigma=model.sigma + k),
        "vol_down": replace(model, sigma=model.sigma - k),
    }

    # Payoff scontati traiettoria per traiettoria in ciascuno scenario.
    # Con CRN tutti usano lo stesso seme -> la traiettoria i-esima parte dagli stessi shock Z
    X = {}
    for i, (name, bumped_model) in enumerate(scenarios.items()):
        run_seed = seed if common_random_numbers else seed + i
        X[name] = discounted_payoffs(
            bumped_model, instrument, n_paths=n_paths, n_steps=n_steps, seed=run_seed
        )

    # Differenze finite applicate a OGNI traiettoria: si ottiene un campione di N stime del greek,
    # la cui media è il greek e la cui dispersione dà l'errore standard. Con CRN ogni differenza
    # confronta lo stesso scenario di mercato; senza CRN confronta scenari scollegati
    delta_samples = (X["spot_up"] - X["spot_down"]) / (2 * h)
    gamma_samples = (X["spot_up"] - 2 * X["base"] + X["spot_down"]) / h**2
    vega_samples = (X["vol_up"] - X["vol_down"]) / (2 * k)

    return GreeksResult(
        price=_estimate(X["base"]),
        delta=_estimate(delta_samples),
        gamma=_estimate(gamma_samples),
        vega=_estimate(vega_samples),
        common_random_numbers=common_random_numbers,
    )
