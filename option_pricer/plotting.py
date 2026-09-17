"""
Stile grafico comune a tutti i notebook: figure in stile accademico.

Convenzioni da articolo scientifico o libro di testo:
  - font serif STIX (quello delle riviste scientifiche), con le formule nello stesso stile
  - riquadro completo, tacche rivolte verso l'interno su tutti i lati, niente griglia
  - scala di grigi: le serie si distinguono per tipo di linea e marcatore,
    così le figure restano leggibili anche stampate in bianco e nero
  - nessun titolo dentro il grafico: ogni figura ha una didascalia numerata sotto

È l'unico modulo del package che usa matplotlib, e __init__.py non lo importa:
i motori di pricing non disegnano nulla e restano indipendenti dalla grafica.

Uso tipico, in cima al notebook:
    from option_pricer.plotting import BLACK, add_caption, apply_style
    apply_style()
"""

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

# --- Scala di grigi ----------------------------------------------------------
BLACK = "#000000"   # dati principali, prezzi esatti, riferimenti
DARK = "#4d4d4d"    # seconda serie
MID = "#8c8c8c"     # terza serie, elementi secondari
LIGHT = "#c8c8c8"   # aree in evidenza
FILL = "#ebebeb"    # bande e aree di sfondo

# Fino a tre serie distinguibili senza colore: linea continua, tratteggiata, tratto-punto
SERIES = [
    dict(color=BLACK, linestyle="-", marker="o"),
    dict(color=DARK, linestyle="--", marker="s"),
    dict(color=MID, linestyle="-.", marker="^"),
]


def _enable_retina() -> None:
    # Nei notebook (Colab, Jupyter, VS Code) mostra le figure a doppia risoluzione: testo nitido.
    # Fuori da un notebook, ad esempio in uno script, non c'è nulla da fare
    try:
        from IPython import get_ipython
        from matplotlib_inline.backend_inline import set_matplotlib_formats
    except ImportError:
        return
    if get_ipython() is not None:
        set_matplotlib_formats("retina")


def apply_style() -> None:
    """Imposta font, assi, tacche e risoluzione per tutti i grafici successivi."""
    plt.rcParams.update({
        # Testo e formule: STIX è incluso in matplotlib, quindi è disponibile ovunque (anche su Colab)
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.formatter.use_mathtext": True,   # numeri sugli assi nello stesso font delle formule (es. 10²)
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11,                  # usato solo per le etichette dei pannelli: (a), (b)
        "axes.titleweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        # Figura
        "figure.figsize": (6.8, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        # Riquadro completo, senza griglia
        "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        # Tacche verso l'interno su tutti e quattro i lati, con tacche secondarie
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        # Linee sottili e legenda con cornice squadrata, come in LaTeX
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": BLACK,
        "legend.framealpha": 1.0,
        "legend.handlelength": 2.2,
        "patch.linewidth": 0.8,
    })
    _enable_retina()


def _wrap_to_width(text: str, max_width: float, fontsize: float) -> str:
    # Va a capo misurando la larghezza REALE di ogni riga (in punti tipografici), formule comprese:
    # contare i caratteri sbaglierebbe, perché "$\sigma$" occupa 8 caratteri ma un solo simbolo
    prop = FontProperties(size=fontsize)
    measure = TextToPath()

    def width(line: str) -> float:
        return measure.get_text_width_height_descent(line, prop, ismath="$" in line)[0]

    lines, current = [], ""
    for word in text.split(" "):
        candidate = f"{current} {word}" if current else word
        if current and width(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def add_caption(fig, number: int, text: str, fontsize: float = 10) -> None:
    """
    Didascalia numerata sotto la figura, come in un articolo: "Figura 1. Testo...".

    Il testo va a capo da solo in base alla larghezza della figura.
    Le formule tra $...$ non devono contenere spazi (usare \\, per lo spazio sottile).
    """
    max_width = 0.97 * fig.get_figwidth() * 72  # larghezza utile in punti (1 pollice = 72 punti)
    body = _wrap_to_width(f"Figura {number}. {text}", max_width, fontsize)
    # Coordinate della figura: x = 0 bordo sinistro, y = 0 bordo inferiore; va="top" mette il testo SOTTO
    fig.text(0.01, -0.01, body, ha="left", va="top", fontsize=fontsize, transform=fig.transFigure)


def math_int(value: int) -> str:
    """Numero intero in formato matematico con spazio sottile tra le migliaia: 200000 -> $200\\,000$."""
    return "$" + f"{value:,}".replace(",", r"\,") + "$"
