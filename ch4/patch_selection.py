"""
Problema 4.28 (Puterman, Markov Decision Processes) -- MLOB 6.

Un animal elige en cada periodo una de 3 parcelas donde refugiarse y
alimentarse. Cada parcela tiene un riesgo de depredacion, una probabilidad
de encontrar comida y la energia que esa comida aporta. Buscar alimento
cuesta siempre 1 unidad de energia. Si la energia cae por debajo de 4 el
animal muere de inanicion; la capacidad maxima es 10. Se busca la politica
de eleccion de parcela que maximiza la probabilidad de sobrevivir 20
periodos.

Resuelto como proceso de decision markoviano de horizonte finito
(ecuacion de Bellman, induccion hacia atras) y validado con simulacion
Monte Carlo.
"""

import random

XMIN, XMAX, T = 4, 10, 20          # energia minima viva, maxima, horizonte
PATCHES = {                        # p: riesgo depredacion, f: prob. comida, e: energia si hay comida
    1: dict(p=0.000, f=0.0, e=0),
    2: dict(p=0.004, f=0.4, e=3),
    3: dict(p=0.02,  f=0.6, e=5),
}
STATES = list(range(XMIN, XMAX + 1))


# ---------- 1. Programacion dinamica (induccion hacia atras) ----------
def solve():
    V = {0: {x: 1.0 for x in STATES}}      # V(x,0)=1: ya sobrevivio todo el horizonte
    policy = {}

    def S(y, t):                            # energia truncada en XMAX; muerte si y<XMIN
        y = min(y, XMAX)
        return V[t][y] if y >= XMIN else 0.0

    for t in range(1, T + 1):
        V[t], policy[t] = {}, {}
        for x in STATES:
            best_val, best_i = -1, None
            for i, par in PATCHES.items():
                p, f, e = par['p'], par['f'], par['e']
                # ley de probabilidad total: sobrevive depredacion, y (comida | sin comida)
                val = (1 - p) * (f * S(x - 1 + e, t - 1) + (1 - f) * S(x - 1, t - 1))
                if val > best_val:
                    best_val, best_i = val, i
            V[t][x], policy[t][x] = best_val, best_i
    return V, policy


# ---------- 2. Simulacion Monte Carlo (validacion) ----------
def simulate_one(x0, choose, rng):
    x, t_left = x0, T
    while t_left > 0:
        i = choose(x, t_left)
        par = PATCHES[i]
        if rng.random() < par['p']:
            return False                     # muerte por depredacion
        x -= 1                               # costo de forrajear
        if rng.random() < par['f']:
            x = min(x + par['e'], XMAX)       # encontro comida
        if x < XMIN:
            return False                     # muerte por inanicion
        t_left -= 1
    return True


def validate(V, policy, n=200_000, seed=42):
    rng = random.Random(seed)
    print(f"\nValidacion Monte Carlo (N={n} trayectorias por x0):")
    print(f"{'x0':>4} {'teoria':>9} {'simulacion':>11} {'IC 95%':>18}")
    for x0 in STATES:
        survivors = sum(simulate_one(x0, lambda x, t: policy[t][x], rng) for _ in range(n))
        p_hat = survivors / n
        se = (p_hat * (1 - p_hat) / n) ** 0.5
        ci = 1.96 * se
        print(f"{x0:>4} {V[T][x0]:>9.4f} {p_hat:>11.4f}   [{p_hat-ci:.4f}, {p_hat+ci:.4f}]")


# ---------- 3. Comparacion con politicas alternativas (no optimas) ----------
def compare_policies(policy, n=100_000, seed=7):
    rng = random.Random(seed)
    alt_policies = {
        "optima (DP)":       lambda x, t: policy[t][x],
        "siempre parcela 1": lambda x, t: 1,
        "siempre parcela 2": lambda x, t: 2,
        "siempre parcela 3": lambda x, t: 3,
        "miope (max. energia esperada)": lambda x, t: max(
            PATCHES,
            key=lambda i: (1 - PATCHES[i]['p']) * PATCHES[i]['f'] * PATCHES[i]['e']
                          - PATCHES[i]['p'] * 5,
        ),
    }
    print(f"\nComparacion con politicas alternativas (N={n} trayectorias por punto):")
    print(f"{'politica':>32} " + " ".join(f"x0={x0:>2}" for x0 in STATES))
    for name, choose in alt_policies.items():
        row = [sum(simulate_one(x0, choose, rng) for _ in range(n)) / n for x0 in STATES]
        print(f"{name:>32} " + " ".join(f"{v:6.4f}" for v in row))


if __name__ == "__main__":
    V, policy = solve()

    print("Politica optima d*(x, t=20) y probabilidad de sobrevivir 20 periodos:")
    for x0 in STATES:
        print(f"  x0={x0:>2}: d*(x0,20)=parcela {policy[T][x0]}   P(sobrevivir 20 periodos) = {V[T][x0]:.4f}")

    validate(V, policy)
    compare_policies(policy)
