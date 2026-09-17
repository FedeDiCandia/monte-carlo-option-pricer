"""
Validazione del motore Monte Carlo contro la formula chiusa di Black-Scholes.

Logica: dove esiste una soluzione esatta, il Monte Carlo deve ritrovarla entro il
proprio errore statistico. Se supera il test su call e put europee, possiamo fidarci
del motore anche sui contratti per cui una formula chiusa non esiste.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .black_scholes import price_black_scholes
from .monte_carlo import MCResult, price_monte_carlo

# Scarti sotto questa soglia sono arrotondamento in virgola mobile (~1e-13), non errore statistico.
# Contano solo nel caso degenere sigma = 0, dove l'intervallo di confidenza ha ampiezza quasi nulla
_ROUNDOFF_TOL = 1e-10


@dataclass(frozen=True)
class ValidationResult:
    """Confronto tra una stima Monte Carlo e il prezzo esatto di Black-Scholes."""

    mc: MCResult     # stima Monte Carlo, con errore standard e intervallo di confidenza
    bs_price: float  # prezzo esatto di Black-Scholes

    @property
    def error(self) -> float:
        # Scarto con segno: > 0 se il Monte Carlo sovrastima il prezzo vero
        return self.mc.price - self.bs_price

    @property
    def z_score(self) -> float:
        # Scarto misurato in errori standard. Se il motore è corretto, per il TLC z ~ N(0, 1):
        # |z| < 1.96 nel 95% dei casi, |z| > 3 solo nello 0.3% -> un |z| grande segnala un bug
        if abs(self.error) <= _ROUNDOFF_TOL:
            return 0.0
        if self.mc.std_error == 0:
            # Stima deterministica ma diversa da BS: nessuna fluttuazione statistica può spiegarla
            return math.copysign(math.inf, self.error)
        return self.error / self.mc.std_error

    @property
    def within_ci(self) -> bool:
        # Il prezzo esatto cade dentro l'intervallo di confidenza del Monte Carlo?
        return abs(self.error) <= _ROUNDOFF_TOL or self.mc.ci_low <= self.bs_price <= self.mc.ci_high

    def __str__(self) -> str:
        verdict = "dentro" if self.within_ci else "FUORI"
        return (
            f"MC {self.mc.price:.4f} ± {self.mc.std_error:.4f}  |  BS {self.bs_price:.4f}  |  "
            f"scarto {self.error:+.4f} = {self.z_score:+.2f} SE  |  "
            f"BS {verdict} l'IC {self.mc.confidence:.0%}"
        )


def compare_with_black_scholes(
    model,
    instrument,
    n_paths: int = 100_000,
    n_steps: int = 1,
    seed: int | None = None,
    confidence: float = 0.95,
) -> ValidationResult:
    """Prezza lo stesso contratto, con gli stessi parametri, con entrambi i motori."""
    return ValidationResult(
        # Black-Scholes per primo: se il contratto non ha formula chiusa l'errore arriva subito, senza simulare
        bs_price=price_black_scholes(model, instrument),
        mc=price_monte_carlo(
            model, instrument, n_paths=n_paths, n_steps=n_steps, seed=seed, confidence=confidence
        ),
    )


def convergence_study(
    model,
    instrument,
    n_values: Sequence[int],
    seed: int | None = None,
    confidence: float = 0.95,
) -> list[MCResult]:
    """
    Ripete il pricing Monte Carlo per ogni N in `n_values`, con simulazioni INDIPENDENTI.

    Ogni N usa un seme diverso (seed, seed+1, ...), quindi le stime non condividono
    scenari: ciascun punto mostra l'errore tipico che si avrebbe davvero usando
    quel numero di traiettorie. Allungare una sola simulazione darebbe invece una
    curva liscia, ma con punti fortemente correlati tra loro.
    """
    results = []
    for i, n in enumerate(n_values):
        # Con seed=None ogni run è casuale; altrimenti semi distinti ma riproducibili
        run_seed = None if seed is None else seed + i
        results.append(
            price_monte_carlo(model, instrument, n_paths=int(n), seed=run_seed, confidence=confidence)
        )
    return results
