"""
Induccion hacia atras aplicada al Problema 4.4 -- asignacion secuencial de
un recurso M entre N periodos (Puterman, seccion 4.6.3; MLOB p.2-3).

Estado s = recurso restante. En cada epoca t=1,...,N-1 se elige una accion
a in [0,s] (cuanto consumir) y se recibe g(a); en la epoca terminal N se
consume todo lo que queda y se recibe g(s). Si g es concava con maximo en
[0,M], la politica optima es d*_t(s) = s/(N-t+1): repartir lo que queda en
partes iguales entre los periodos restantes, lo que produce el patron de
consumo constante a_t = M/N para todo t.

El PDF resuelve el caso general por induccion hacia atras y prueba
u*_{N-t+1}(s) = t*g(s/t),  d*_{N-t+1}(s) = s/t   para t = 1,...,N.

Este script resuelve una version discretizada del mismo problema (estado
continuo en una malla) y compara contra esa forma cerrada.
"""


def solve_allocation(g, M, N, ngrid=400):
    """
    g:      funcion de recompensa por periodo (concava en [0,M])
    M:      recurso total a repartir
    N:      numero de periodos
    ngrid:  resolucion de la malla en [0,M]

    Regresa (grid, V, policy). V[t][s] y policy[t][s] estan definidos para
    t=1,...,N-1 (la epoca N consume todo el remanente automaticamente).
    """
    grid = [M * k / ngrid for k in range(ngrid + 1)]

    def nearest(x):
        return min(grid, key=lambda g_: abs(g_ - x))

    V = {N: {s: g(s) for s in grid}}
    policy = {}
    for t in range(N - 1, 0, -1):
        V[t], policy[t] = {}, {}
        for s in grid:
            best_val, best_a = -float("inf"), None
            for a in grid:
                if a > s + 1e-9:
                    break
                val = g(a) + V[t + 1][nearest(s - a)]
                if val > best_val:
                    best_val, best_a = val, a
            V[t][s], policy[t][s] = best_val, best_a
    return grid, V, policy


def nearest_val(table, x, grid):
    k = min(grid, key=lambda kk: abs(kk - x))
    return table[k]


def optimal_trajectory(g, M, N, policy, grid):
    """Traza s_t, a_t siguiendo la politica optima desde s_1 = M."""
    traj, s = [], M
    for t in range(1, N + 1):
        if t == N:
            traj.append((t, s, s, g(s)))     # consume todo lo que queda
            break
        a = nearest_val(policy[t], s, grid)
        traj.append((t, s, a, g(a)))
        s -= a
    return traj


if __name__ == "__main__":
    import math

    g = math.sqrt              # cualquier g concava sirve; sqrt es solo un ejemplo
    M, N = 10.0, 4

    grid, V, policy = solve_allocation(g, M, N, ngrid=200)

    print("Comparacion forma cerrada u*_{N-t+1}(s) = t*g(s/t)  vs.  DP discretizada, en s=M:")
    print(f"{'epoca':>6} {'u* numerico':>12} {'u* cerrado':>12} {'a* numerico':>12} {'a* cerrado':>11}")
    for epoch in range(1, N + 1):
        t = N - epoch + 1
        closed_u, closed_a = t * g(M / t), M / t
        numeric_u = nearest_val(V[epoch], M, grid)
        numeric_a = nearest_val(policy[epoch], M, grid) if epoch in policy else M
        print(f"{epoch:>6} {numeric_u:>12.4f} {closed_u:>12.4f} {numeric_a:>12.4f} {closed_a:>11.4f}")

    print(f"\nTrayectoria optima desde s_1={M} (reparto igualitario esperado: M/N = {M/N}):")
    for t, s, a, r in optimal_trajectory(g, M, N, policy, grid):
        print(f"  t={t}: s_t={s:.4f}  a_t*={a:.4f}  g(a_t*)={r:.4f}")

    print(
        "\nNota: el enunciado del PDF dice 'consumir M/(N-1) unidades cada periodo', "
        "pero su propia demostracion (u*_{N-t+1}(s)=t*g(s/t), d*=s/t) y esta DP "
        "numerica coinciden en que la asignacion optima es constante e igual a "
        "M/N -- 'N-1' en el enunciado parece un error de transcripcion."
    )
