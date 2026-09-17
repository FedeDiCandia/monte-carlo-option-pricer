"""
Modelli per la dinamica del sottostante.

Un "modello" risponde a UNA sola domanda: come si muove il prezzo nel tempo?
Non sa nulla di opzioni, strike o payoff: quello è compito degli strumenti
(instruments.py). Questa separazione permette di cambiare modello senza
toccare i contratti, e viceversa.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)  # frozen=True: i parametri non si possono modificare per errore dopo la creazione
class GBM:
    """
    Geometric Brownian Motion sotto la misura risk-neutral Q:

        dS_t = r * S_t * dt + sigma * S_t * dW_t

    Soluzione esatta (via lemma di Itô applicato a ln S):

        S_{t+dt} = S_t * exp( (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z ),  Z ~ N(0, 1)

    Parametri
    ---------
    S0    : prezzo spot del sottostante oggi (t = 0)
    r     : tasso risk-free annuo, capitalizzazione continua (0.05 = 5%)
    sigma : volatilità annua dei rendimenti logaritmici (0.20 = 20%)
    """

    S0: float
    r: float
    sigma: float

    def __post_init__(self):
        # Controlli minimi di coerenza: un prezzo <= 0 o una volatilità negativa non hanno senso
        if self.S0 <= 0:
            raise ValueError(f"S0 deve essere > 0, ricevuto {self.S0}")
        if self.sigma < 0:
            raise ValueError(f"sigma deve essere >= 0, ricevuto {self.sigma}")

    def simulate_paths(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Simula n_paths traiettorie del prezzo da t = 0 a t = T.

        Restituisce una matrice di forma (n_paths, n_steps + 1):
          - ogni RIGA è una traiettoria (uno "scenario" possibile del futuro)
          - ogni COLONNA è un istante temporale; la colonna 0 è S0, l'ultima è S_T

        Nota: per la GBM lo schema usato è ESATTO per qualunque dt (non è
        un'approssimazione di Euler), quindi per un'opzione europea basta
        n_steps = 1. Più passi servono solo per payoff che dipendono dal
        percorso (es. opzioni asiatiche o barriera).
        """
        if T <= 0 or n_steps < 1 or n_paths < 1:
            raise ValueError("Servono T > 0, n_steps >= 1, n_paths >= 1")

        # Ampiezza di ogni passo temporale, in anni
        dt = T / n_steps

        # Drift del LOG-prezzo per passo: (r - 0.5*sigma^2)*dt.
        # Il termine -0.5*sigma^2 è la correzione di Itô: garantisce E[S_T] = S0 * e^(rT)
        drift = (self.r - 0.5 * self.sigma**2) * dt

        # Deviazione standard del log-rendimento per passo: la varianza cresce
        # linearmente col tempo, quindi la dev. std cresce come sqrt(dt)
        diffusion = self.sigma * np.sqrt(dt)

        # Shock casuali: una N(0,1) indipendente per ogni traiettoria e ogni passo
        Z = rng.standard_normal(size=(n_paths, n_steps))

        # Log-rendimento di ogni singolo passo: parte deterministica + parte casuale
        log_returns = drift + diffusion * Z

        # I log-rendimenti si SOMMANO nel tempo: ln(S_t / S0) = somma cumulata dei log-rendimenti
        cumulative_log_returns = np.cumsum(log_returns, axis=1)

        # Aggiungo una colonna di zeri in testa, così la colonna 0 corrisponde a t = 0 (ln(S0/S0) = 0)
        cumulative_log_returns = np.hstack([np.zeros((n_paths, 1)), cumulative_log_returns])

        # Torno dai log-prezzi ai prezzi: S_t = S0 * exp(somma dei log-rendimenti) -> sempre > 0
        return self.S0 * np.exp(cumulative_log_returns)
