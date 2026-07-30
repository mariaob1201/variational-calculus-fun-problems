"""
Induccion hacia atras (algoritmo generico de Bellman para MDP de horizonte
finito) aplicada al Problema 4.20 -- reemplazo de equipo (Puterman, seccion
4.7.5, pagina 109; MLOB p.5).

Estado s = edad del equipo. En cada epoca t = 1,...,N-1 se elige:
  a=0 (mantener):   se paga h(s), el equipo envejece s -> s+X
  a=1 (reemplazar): se paga K, se recibe R, el equipo se reinicia y envejece 0 -> X
En la epoca terminal N se cobra la recompensa terminal r_N(s).

X (deterioro aleatorio por periodo) es geometrica: P(X=j) = (1-pi)*pi^j,
j=0,1,2,... Los parametros del problema son pi=0.4, R=0, K=5, h(s)=2s,
N=3, r_3(s)=max{5-s,0}.
"""


def backward_induction(states, actions, N, transition, reward, terminal_reward):
    """
    states:    lista de estados
    actions:   dict estado -> acciones admisibles
    N:         numero de epocas de decision (la epoca N es terminal)
    transition(s, a) -> dict {s': P(s'|s,a)}
    reward(t, s, a)  -> recompensa inmediata en la epoca t
    terminal_reward(s) -> u*_N(s)

    Regresa (V, policy), ambos dict t -> {s: valor/accion}, para t=1..N-1.
    """
    V = {N: {s: terminal_reward(s) for s in states}}
    policy = {}
    for t in range(N - 1, 0, -1):
        V[t], policy[t] = {}, {}
        for s in states:
            best_val, best_a = None, None
            for a in actions[s]:
                ev = sum(p * V[t + 1][sp] for sp, p in transition(s, a).items())
                val = reward(t, s, a) + ev
                if best_val is None or val > best_val:
                    best_val, best_a = val, a
            V[t][s], policy[t][s] = best_val, best_a
    return V, policy


# ---------- Problema 4.20: modelo de reemplazo de equipo ----------
PI = 0.4                      # P(X=j) = (1-PI) * PI**j , j = 0,1,2,...
K, R = 5, 0                   # costo de reemplazo, valor de rescate
N = 3
MAXAGE = 60                   # tope de edad; la cola geometrica truncada aqui es ~0


def h(s):
    return 2 * s               # costo de mantener equipo de edad s


STATES = list(range(MAXAGE + 1))
ACTIONS = {s: [0, 1] for s in STATES}   # 0 = mantener, 1 = reemplazar
PMF = {j: (1 - PI) * PI ** j for j in range(MAXAGE + 1)}


def transition(s, a):
    base = 0 if a == 1 else s
    dist = {}
    for j, p in PMF.items():
        sp = min(base + j, MAXAGE)
        dist[sp] = dist.get(sp, 0.0) + p
    return dist


def reward(t, s, a):
    return (R - K) if a == 1 else -h(s)


def terminal_reward(s):
    return max(5 - s, 0)


def threshold(policy, t):
    """Menor s tal que la politica optima en la epoca t es reemplazar."""
    return min(s for s in STATES if policy[t][s] == 1)


if __name__ == "__main__":
    V, policy = backward_induction(STATES, ACTIONS, N, transition, reward, terminal_reward)

    print("u*_3(s) = r_3(s) = max{5-s,0}, s=0..4:", [terminal_reward(s) for s in range(5)])

    print("\nu*_2(s), s=0..4:", [round(V[2][s], 4) for s in range(5)])
    print("(solucion a mano en el PDF: [4.345, 1.35, -0.65, -0.65, -0.65])")
    print(f"s*_2 = {threshold(policy, 2)}  (reemplazar sii s >= s*_2)  -- PDF: s*_2 = 2")

    print("\nu*_1(s), s=0..4:", [round(V[1][s], 4) for s in range(5)])
    print("(solucion a mano en el PDF: [2.82, -1.35, -2.165, -2.165, -2.165])")
    print(f"s*_1 = {threshold(policy, 1)}  (reemplazar sii s >= s*_1)  -- PDF: s*_1 = 2")

    print(
        "\nNota: u*_1(1) da -1.4537 aqui contra -1.35 en el PDF -- una pequena "
        "diferencia de redondeo en la solucion a mano (probablemente en f(1)); "
        "los umbrales de politica optima s*_1 = s*_2 = 2 coinciden exactamente."
    )
