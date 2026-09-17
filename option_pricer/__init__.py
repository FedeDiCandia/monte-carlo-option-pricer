"""
option_pricer — pricing di opzioni con Monte Carlo.

Architettura a strati indipendenti:
  models.py         -> COME si muove il sottostante   (GBM)
  instruments.py    -> COSA paga il contratto         (europee, asiatiche, barriere)
  monte_carlo.py    -> prezzo STIMATO numericamente   (media dei payoff scontati)
  black_scholes.py  -> prezzo ESATTO in formula chiusa (solo europee sotto GBM)
  validation.py     -> CONFRONTO tra i due motori     (Monte Carlo vs Black-Scholes)
  greeks.py         -> SENSIBILITÀ del prezzo         (differenze finite con CRN)
  plotting.py       -> ASPETTO dei grafici            (non importato qui: il pricing non dipende da matplotlib)
"""

from .black_scholes import (
    black_scholes_call,
    black_scholes_greeks,
    black_scholes_put,
    d1_d2,
    price_black_scholes,
)
from .greeks import Estimate, GreeksResult, compute_greeks
from .instruments import AsianOption, BarrierOption, EuropeanCall, EuropeanPut
from .models import GBM
from .monte_carlo import MCResult, discounted_payoffs, price_monte_carlo
from .validation import ValidationResult, compare_with_black_scholes, convergence_study

__all__ = [
    "GBM",
    "EuropeanCall",
    "EuropeanPut",
    "AsianOption",
    "BarrierOption",
    "MCResult",
    "discounted_payoffs",
    "price_monte_carlo",
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_greeks",
    "d1_d2",
    "price_black_scholes",
    "ValidationResult",
    "compare_with_black_scholes",
    "convergence_study",
    "Estimate",
    "GreeksResult",
    "compute_greeks",
]
