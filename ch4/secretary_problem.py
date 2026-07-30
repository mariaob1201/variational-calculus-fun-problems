"""
Problema 4.10 -- el problema de la secretaria, variante de rango esperado
(Chow, Robbins, Samuels 1964; MLOB p.3). En el PDF esta resuelto solo hasta
la formula de la utilidad esperada U(r,j) ("NO HAS TERMINADO"); este script
completa la induccion hacia atras que el enunciado deja pendiente.

Se observan n candidatos uno a la vez, en orden aleatorio, y solo se conoce
el rango relativo del candidato actual entre los ya vistos (no su rango
verdadero). Al detenerse en la posicion r aceptando al candidato cuyo rango
relativo es j, la utilidad recibida es u = n - i (i = rango verdadero, 1 =
mejor), y su valor esperado es:

    U(r,j) = n - j*(n+1)/(r+1)

(formula que el PDF deriva y deja ahi). El objetivo es maximizar la
utilidad esperada, equivalente a minimizar el rango esperado del candidato
aceptado.
"""

import random


def solve_secretary(n):
    """
    Induccion hacia atras sobre la posicion r = n,...,1.

    V[r][j] = valor optimo de estar en la posicion r con rango relativo j
              (mejor entre detenerse ahora y seguir observando).
    W[r]    = valor esperado de continuar despues de la posicion r (no
              depende de j: el rango relativo del candidato r+1 es uniforme
              en {1,...,r+1}, sin importar la historia).

    Regresa (expected_rank, thresholds): el rango esperado del candidato
    aceptado bajo la politica optima, y el umbral s*_r tal que la regla
    optima es "aceptar en r si el rango relativo j <= s*_r".
    """
    def U(r, j):
        return n - j * (n + 1) / (r + 1)

    V = {n: {j: U(n, j) for j in range(1, n + 1)}}   # en r=n hay que aceptar si o si
    W = {}
    for r in range(n - 1, 0, -1):
        W[r] = sum(V[r + 1][jp] for jp in range(1, r + 2)) / (r + 1)
        V[r] = {j: max(U(r, j), W[r]) for j in range(1, r + 1)}

    value = V[1][1]                 # unico estado posible en r=1 (j=1 siempre)
    expected_rank = n - value

    thresholds = {}
    for r in range(1, n):
        thresholds[r] = max((j for j in range(1, r + 1) if U(r, j) >= W[r]), default=0)
    return expected_rank, thresholds


def simulate(n, thresholds, trials, rng):
    """Valida la politica optima con permutaciones aleatorias de los n
    rangos verdaderos, aceptando en la primera posicion r donde el rango
    relativo observado cae dentro del umbral."""
    total = 0
    for _ in range(trials):
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        chosen_rank = perm[-1]                  # si nunca se para antes, se acepta el ultimo
        for r in range(1, n + 1):
            val = perm[r - 1]
            rel_rank = sum(1 for x in perm[:r] if x <= val)
            if r == n or rel_rank <= thresholds.get(r, 0):
                chosen_rank = val
                break
        total += chosen_rank
    return total / trials


CRSZ_CONSTANT = 3.8695          # limite E[rango*] cuando n -> infinito (Chow-Robbins-Samuels-Zabell 1964)


if __name__ == "__main__":
    print(f"{'n':>5} {'E[rango*] teorico':>18} {'simulacion (N=100k)':>20}")
    rng = random.Random(0)
    for n in (10, 20, 50, 100, 500):
        expected_rank, thresholds = solve_secretary(n)
        sim = simulate(n, thresholds, 100_000, rng)
        print(f"{n:>5} {expected_rank:>18.4f} {sim:>20.4f}")
    print(f"\nConstante limite conocida (n->infinito): {CRSZ_CONSTANT}")

    n = 20
    expected_rank, thresholds = solve_secretary(n)
    print(f"\nUmbrales optimos para n={n} (aceptar en la posicion r si rango relativo j <= s*_r):")
    for r in range(1, n):
        print(f"  r={r:2d}: s*_r={thresholds[r]}")
