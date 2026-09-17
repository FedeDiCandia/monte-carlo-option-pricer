"""
Motore di pricing Monte Carlo.

Il motore unisce modello e strumento applicando la risk-neutral valuation:

    Prezzo = e^(-rT) * E^Q[ payoff ]

e approssima il valore atteso con la media campionaria su N scenari simulati.
Non sa che tipo di opzione sta prezzando: chiama solo `instrument.payoff(paths)`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class MCResult:
    """Risultato di un pricing Monte Carlo: la stima E la sua incertezza."""

    price: float       # stima del prezzo (media dei payoff scontati)
    std_error: float   # errore standard della stima = s / sqrt(N)
    ci_low: float      # estremo inferiore dell'intervallo di confidenza
    ci_high: float     # estremo superiore dell'intervallo di confidenza
    confidence: float  # livello di confidenza dell'intervallo (es. 0.95)
    n_paths: int       # numero di traiettorie usate

    def __str__(self) -> str:
        return (
            f"{self.price:.4f} ± {self.std_error:.4f} (SE)  |  "
            f"IC {self.confidence:.0%}: [{self.ci_low:.4f}, {self.ci_high:.4f}]  |  N = {self.n_paths:,}"
        )


def discounted_payoffs(
    model,
    instrument,
    n_paths: int,
    n_steps: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simula le traiettorie e restituisce il vettore (n_paths,) dei payoff SCONTATI a oggi.

    È separata da `price_monte_carlo` perché il vettore completo serve anche
    per analisi diagnostiche (es. grafico di convergenza), non solo la sua media.
    """
    # Generatore di numeri casuali moderno di numpy; con un seed fisso i risultati sono riproducibili
    rng = np.random.default_rng(seed)

    # 1) MODELLO: genera gli scenari futuri del sottostante, matrice (n_paths, n_steps + 1)
    paths = model.simulate_paths(T=instrument.T, n_steps=n_steps, n_paths=n_paths, rng=rng)

    # 2) STRUMENTO: calcola quanto paga il contratto in ciascuno scenario, vettore (n_paths,)
    payoffs = instrument.payoff(paths)

    # 3) SCONTO: il payoff arriva in T, lo riporto a oggi con il fattore risk-free e^(-rT)
    discount_factor = np.exp(-model.r * instrument.T)
    return discount_factor * payoffs


def price_monte_carlo(
    model,
    instrument,
    n_paths: int = 100_000,
    n_steps: int = 1,
    seed: int | None = None,
    confidence: float = 0.95,
) -> MCResult:
    """
    Prezza `instrument` sotto `model` con Monte Carlo.

    Parametri
    ---------
    model      : oggetto con .simulate_paths(...) e .r (es. GBM)
    instrument : oggetto con .payoff(paths) e .T (es. EuropeanCall)
    n_paths    : N, numero di traiettorie simulate
    n_steps    : passi temporali per traiettoria (1 basta per le europee)
    seed       : seme casuale per la riproducibilità (None = casuale)
    confidence : livello dell'intervallo di confidenza
    """
    # Un payoff scontato per ogni scenario: sono N realizzazioni i.i.d. della stessa variabile aleatoria
    X = discounted_payoffs(model, instrument, n_paths=n_paths, n_steps=n_steps, seed=seed)

    # Legge dei grandi numeri: la media campionaria converge al valore atteso vero (= prezzo)
    price = X.mean()

    # Deviazione standard campionaria dei payoff scontati (ddof=1 -> stimatore corretto, divide per N-1)
    sample_std = X.std(ddof=1)

    # Teorema del limite centrale: la media ha dev. std s / sqrt(N) -> questo è l'errore standard
    std_error = sample_std / np.sqrt(n_paths)

    # Quantile della normale standard per l'intervallo bilaterale (0.95 -> z ≈ 1.96)
    z = norm.ppf(0.5 + confidence / 2)

    return MCResult(
        price=float(price),
        std_error=float(std_error),
        ci_low=float(price - z * std_error),
        ci_high=float(price + z * std_error),
        confidence=confidence,
        n_paths=n_paths,
    )
