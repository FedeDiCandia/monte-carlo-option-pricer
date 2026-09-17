"""
Strumenti (i contratti da prezzare).

Uno strumento risponde a UNA sola domanda: dato uno scenario di prezzi,
quanto paga il contratto? Non sa come quei prezzi sono stati generati
(quello è il modello) né come si calcola la media (quello è il motore).

Convenzione comune a tutti gli strumenti:
  - ricevono `paths`, matrice (n_paths, n_steps + 1) prodotta da un modello
  - restituiscono un vettore (n_paths,) con il payoff di ciascuna traiettoria

Grazie a questa interfaccia unica, le opzioni esotiche (che guardano l'intera
traiettoria e non solo l'ultimo prezzo) si aggiungeranno come nuove classi,
senza toccare il motore Monte Carlo.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class EuropeanCall:
    """
    Call europea: diritto (non obbligo) di COMPRARE il sottostante a prezzo K alla data T.

    Payoff a scadenza: max(S_T - K, 0)
      - se S_T > K esercito: compro a K ciò che vale S_T, guadagno S_T - K
      - se S_T <= K non esercito: payoff 0 (non perdo più del premio pagato)

    K : strike (prezzo di esercizio)
    T : scadenza in anni (0.5 = sei mesi)
    """

    K: float
    T: float

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # Un'europea guarda solo il prezzo finale: ultima colonna della matrice
        S_T = paths[:, -1]
        # max elemento per elemento su tutte le traiettorie in un colpo solo (vettorizzato, niente loop)
        return np.maximum(S_T - self.K, 0.0)


@dataclass(frozen=True)
class EuropeanPut:
    """
    Put europea: diritto (non obbligo) di VENDERE il sottostante a prezzo K alla data T.

    Payoff a scadenza: max(K - S_T, 0)
      - se S_T < K esercito: vendo a K ciò che vale S_T, guadagno K - S_T
      - se S_T >= K non esercito: payoff 0
    """

    K: float
    T: float

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # Solo il prezzo finale conta
        S_T = paths[:, -1]
        # Payoff speculare alla call
        return np.maximum(self.K - S_T, 0.0)


# ---------------------------------------------------------------------------
# Opzioni esotiche path-dependent
#
# Il payoff dipende da TUTTA la traiettoria, non solo da S_T: per questo servono
# matrici `paths` con n_steps > 1. Le varianti si moltiplicano (una barriera è
# call/put × up/down × in/out = 8 contratti), quindi la variante è un parametro
# della classe invece di una classe separata per ciascuna combinazione.
# ---------------------------------------------------------------------------

OptionKind = Literal["call", "put"]


def _check_choice(value: str, allowed: tuple[str, ...], field: str) -> None:
    # Un refuso ("Call", "knockout") produrrebbe in silenzio un payoff sbagliato: meglio fermarsi subito
    if value not in allowed:
        raise ValueError(f"{field} deve essere uno tra {allowed}, ricevuto {value!r}")


def _vanilla_payoff(underlying: np.ndarray, K: float, kind: OptionKind) -> np.ndarray:
    # Payoff di call o put applicato a un valore qualsiasi del sottostante (S_T, una media, ...)
    if kind == "call":
        return np.maximum(underlying - K, 0.0)
    return np.maximum(K - underlying, 0.0)


def _require_full_path(paths: np.ndarray, contract: str) -> None:
    # Con n_steps = 1 la matrice ha due sole colonne, S0 e S_T: di ciò che succede nel mezzo
    # non sappiamo nulla, e un payoff path-dependent calcolato così sarebbe sbagliato in silenzio
    n_steps = paths.shape[1] - 1
    if n_steps < 2:
        raise ValueError(
            f"{contract} dipende dall'intera traiettoria: simula con n_steps > 1 "
            f"(ricevuto n_steps = {n_steps})"
        )


@dataclass(frozen=True)
class AsianOption:
    """
    Opzione asiatica a media aritmetica, con strike fisso.

    Payoff a scadenza, con A = media dei prezzi osservati lungo la vita dell'opzione:
      call: max(A - K, 0)        put: max(K - A, 0)

    Le date di osservazione ("fixing") sono i passi della simulazione t1, ..., tn.
    Quindi n_steps NON è solo un dettaglio numerico ma una clausola del contratto:
    n_steps = 252 con T = 1 significa "media delle chiusure giornaliere di un anno".

    Non esiste una formula chiusa: una somma di lognormali non è lognormale.
    """

    K: float
    T: float
    kind: OptionKind

    def __post_init__(self):
        _check_choice(self.kind, ("call", "put"), "kind")

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        _require_full_path(paths, "AsianOption")
        # Media per riga (una per traiettoria) sulle colonne 1..n. La colonna 0 è esclusa:
        # S0 è il prezzo di oggi, non una data di osservazione del contratto
        average = paths[:, 1:].mean(axis=1)
        return _vanilla_payoff(average, self.K, self.kind)


@dataclass(frozen=True)
class BarrierOption:
    """
    Opzione con barriera (knock-out / knock-in), senza rebate.

      direction = "up"   -> barriera sopra S0, toccata se max(S_t) >= barrier
      direction = "down" -> barriera sotto S0, toccata se min(S_t) <= barrier
      knock = "out"      -> l'opzione MUORE se la barriera viene toccata (payoff 0)
      knock = "in"       -> l'opzione NASCE solo se la barriera viene toccata

    Se è attiva a scadenza, paga come l'europea: max(S_T - K, 0) o max(K - S_T, 0).

    Monitoraggio DISCRETO: la barriera è controllata solo nelle date simulate
    (n_steps = 252 -> chiusure giornaliere). Un attraversamento avvenuto tra due
    date non viene visto, quindi anche qui il prezzo dipende da n_steps.

    Parità in-out: knock-in + knock-out = europea, perché su ogni traiettoria
    esattamente una delle due è attiva.
    """

    K: float
    T: float
    barrier: float
    kind: OptionKind
    direction: Literal["up", "down"]
    knock: Literal["in", "out"]

    def __post_init__(self):
        _check_choice(self.kind, ("call", "put"), "kind")
        _check_choice(self.direction, ("up", "down"), "direction")
        _check_choice(self.knock, ("in", "out"), "knock")
        if self.barrier <= 0:
            raise ValueError(f"barrier deve essere > 0, ricevuto {self.barrier}")

    def touched(self, paths: np.ndarray) -> np.ndarray:
        """Vettore booleano (n_paths,): True se la traiettoria ha toccato la barriera."""
        _require_full_path(paths, "BarrierOption")
        # Basta l'estremo della traiettoria: se il punto più alto (o più basso) non raggiunge
        # la barriera, nessun altro punto la raggiunge. La colonna 0 (S0) è inclusa:
        # se S0 è già oltre la barriera, il contratto risulta toccato fin dall'inizio
        if self.direction == "up":
            return paths.max(axis=1) >= self.barrier
        return paths.min(axis=1) <= self.barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        hit = self.touched(paths)

        # Payoff "se l'opzione fosse attiva": dipende solo dal prezzo finale
        vanilla = _vanilla_payoff(paths[:, -1], self.K, self.kind)

        # Knock-out: attiva se la barriera NON è stata toccata. Knock-in: attiva solo se toccata
        alive = ~hit if self.knock == "out" else hit

        # np.where: per ogni traiettoria prende il payoff europeo se attiva, altrimenti 0
        return np.where(alive, vanilla, 0.0)
