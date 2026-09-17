"""
Formula chiusa di Black-Scholes (1973) per call e put europee.

È un secondo motore di pricing, alternativo a monte_carlo.py:
  monte_carlo.py   -> stima NUMERICA: vale per qualunque payoff, ma ha un errore statistico
  black_scholes.py -> soluzione ESATTA: nessun errore, ma vale solo per europee sotto GBM

I due motori ricevono gli stessi oggetti (model, instrument), quindi si possono
confrontare direttamente: è esattamente ciò che fa validation.py.
"""

import numpy as np
from scipy.stats import norm

from .instruments import EuropeanCall, EuropeanPut
from .models import GBM


def _check_inputs(S0: float, K: float, T: float, sigma: float) -> None:
    # ln(S0/K) richiede prezzi positivi; scadenza e volatilità negative non hanno senso finanziario
    if S0 <= 0 or K <= 0:
        raise ValueError(f"Servono S0 > 0 e K > 0, ricevuti S0={S0}, K={K}")
    if T <= 0:
        raise ValueError(f"T deve essere > 0, ricevuto {T}")
    if sigma < 0:
        raise ValueError(f"sigma deve essere >= 0, ricevuto {sigma}")


def d1_d2(S0: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """
    I due argomenti della normale cumulata N(.) nella formula di Black-Scholes:

        d2 = [ ln(S0/K) + (r - 0.5*sigma^2)*T ] / (sigma*sqrt(T))
        d1 = d2 + sigma*sqrt(T)

    Interpretazione: N(d2) è la probabilità risk-neutral che S_T > K,
    cioè che la call venga esercitata a scadenza.
    """
    _check_inputs(S0, K, T, sigma)
    if sigma == 0:
        raise ValueError("d1 e d2 non sono definiti per sigma = 0 (divisione per zero)")

    # Deviazione standard di ln(S_T): è il "metro" con cui misuriamo la distanza da K
    vol_sqrt_T = sigma * np.sqrt(T)

    # Numeratore: distanza tra il log-prezzo atteso a scadenza, ln S0 + (r - 0.5*sigma^2)*T,
    # e ln K (stesso drift con correzione di Itô usato dalla GBM in models.py).
    # Dividendo per vol_sqrt_T la distanza diventa "numero di deviazioni standard"
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / vol_sqrt_T

    # d1 sta esattamente una deviazione standard sopra d2
    d1 = d2 + vol_sqrt_T

    return float(d1), float(d2)


def black_scholes_call(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Prezzo di una call europea:   C = S0 * N(d1) - K * e^(-rT) * N(d2)

    Lettura intuitiva: valore attuale di ciò che ricevo esercitando (il sottostante)
    meno valore attuale di ciò che pago (lo strike, pagato solo con probabilità N(d2)).
    """
    _check_inputs(S0, K, T, sigma)

    # Valore attuale dello strike: K pagato in T vale K * e^(-rT) oggi
    K_disc = K * np.exp(-r * T)

    if sigma == 0:
        # Caso limite senza incertezza: S_T = S0 * e^(rT) con certezza,
        # quindi il prezzo è il payoff (già noto oggi) scontato: max(S0 - K*e^(-rT), 0)
        return float(max(S0 - K_disc, 0.0))

    d1, d2 = d1_d2(S0, K, T, r, sigma)

    # norm.cdf(x) = N(x), funzione di ripartizione della normale standard: P(Z <= x)
    return float(S0 * norm.cdf(d1) - K_disc * norm.cdf(d2))


def black_scholes_put(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Prezzo di una put europea:   P = K * e^(-rT) * N(-d2) - S0 * N(-d1)

    Speculare alla call: ricevo lo strike e consegno il sottostante.
    N(-d2) = 1 - N(d2) è la probabilità risk-neutral che la put venga esercitata (S_T < K).
    """
    _check_inputs(S0, K, T, sigma)

    K_disc = K * np.exp(-r * T)

    if sigma == 0:
        # Stesso caso limite deterministico della call, con payoff rovesciato
        return float(max(K_disc - S0, 0.0))

    d1, d2 = d1_d2(S0, K, T, r, sigma)
    return float(K_disc * norm.cdf(-d2) - S0 * norm.cdf(-d1))


# Quali contratti hanno una formula chiusa, e quale. Un contratto nuovo senza formula
# (es. un'esotica) semplicemente non compare qui: price_black_scholes lo rifiuta.
_FORMULAS = {
    EuropeanCall: black_scholes_call,
    EuropeanPut: black_scholes_put,
}


def _check_supported(model, instrument) -> None:
    # Black-Scholes deriva dall'ipotesi che il sottostante segua una GBM: con un altro modello la formula è sbagliata
    if not isinstance(model, GBM):
        raise TypeError(f"Black-Scholes presuppone un modello GBM, ricevuto {type(model).__name__}")
    if type(instrument) not in _FORMULAS:
        raise TypeError(
            f"Nessuna formula chiusa per {type(instrument).__name__}: usa price_monte_carlo"
        )


def price_black_scholes(model: GBM, instrument) -> float:
    """
    Prezza `instrument` sotto `model` con la formula chiusa di Black-Scholes.

    Stessa interfaccia di price_monte_carlo(model, instrument): per passare da un
    motore all'altro basta cambiare funzione, non gli oggetti.
    Restituisce un float (non un MCResult): il prezzo è esatto, non c'è errore standard.
    """
    _check_supported(model, instrument)
    formula = _FORMULAS[type(instrument)]
    return formula(S0=model.S0, K=instrument.K, T=instrument.T, r=model.r, sigma=model.sigma)


def black_scholes_greeks(model: GBM, instrument) -> dict[str, float]:
    """
    Greeks esatti di call e put europee, derivando analiticamente la formula:

      Delta = ∂V/∂S0      call: N(d1)      put: N(d1) - 1
      Gamma = ∂²V/∂S0²    phi(d1) / (S0 * sigma * sqrt(T))    (uguale per call e put)
      Vega  = ∂V/∂sigma   S0 * phi(d1) * sqrt(T)              (uguale per call e put)

    phi = densità della normale standard. Sono il riferimento per validare i greeks
    Monte Carlo di greeks.py, come price_black_scholes valida i prezzi.
    """
    _check_supported(model, instrument)
    S0, sigma, T = model.S0, model.sigma, instrument.T
    d1, _ = d1_d2(S0, instrument.K, T, model.r, sigma)

    # Delta della call = N(d1): quante unità di sottostante servono per replicare l'opzione
    delta = norm.cdf(d1) if isinstance(instrument, EuropeanCall) else norm.cdf(d1) - 1.0

    # Gamma e vega coincidono per call e put: per la put-call parity C - P = S0 - K*e^(-rT),
    # una differenza lineare in S0 (derivata seconda nulla) e indipendente da sigma
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}
