"""
El problema del agente viajero (TSP) resuelto con las mismas dos
herramientas usadas en el resto de ch4/: programacion dinamica exacta
(Held-Karp, una recursion de Bellman -- pero sobre subconjuntos visitados,
no sobre el tiempo) y una heuristica, comparadas sobre datos reales de
las principales ciudades de EE.UU.

Datos: las 50 ciudades mas pobladas de EE.UU. (poblacion, latitud,
longitud), tomadas de plotly/datasets (us-cities-top-1k.csv, a su vez
derivado de datos censales de EE.UU.).
https://github.com/plotly/datasets/blob/master/us-cities-top-1k.csv

Held-Karp resuelve exacto pero es O(2^n * n^2) en tiempo y O(2^n * n) en
memoria -- en Python puro es practico hasta n~18-20. Para las 50 ciudades
completas se usa una heuristica (vecino mas cercano + 2-opt, con reinicios
multiples), que no garantiza el optimo.
"""

import math
import time

# (ciudad, estado, poblacion, lat, lon) -- top 50 ciudades de EE.UU. por poblacion
US_CITIES = [
    ("New York", "New York", 8405837, 40.7128, -74.0059),
    ("Los Angeles", "California", 3884307, 34.0522, -118.2437),
    ("Chicago", "Illinois", 2718782, 41.8781, -87.6298),
    ("Houston", "Texas", 2195914, 29.7604, -95.3698),
    ("Philadelphia", "Pennsylvania", 1553165, 39.9526, -75.1652),
    ("Phoenix", "Arizona", 1513367, 33.4484, -112.0740),
    ("San Antonio", "Texas", 1409019, 29.4241, -98.4936),
    ("San Diego", "California", 1355896, 32.7157, -117.1611),
    ("Dallas", "Texas", 1257676, 32.7767, -96.7970),
    ("San Jose", "California", 998537, 37.3382, -121.8863),
    ("Austin", "Texas", 885400, 30.2672, -97.7431),
    ("Indianapolis", "Indiana", 843393, 39.7684, -86.1581),
    ("Jacksonville", "Florida", 842583, 30.3322, -81.6557),
    ("San Francisco", "California", 837442, 37.7749, -122.4194),
    ("Columbus", "Ohio", 822553, 39.9612, -82.9988),
    ("Charlotte", "North Carolina", 792862, 35.2271, -80.8431),
    ("Fort Worth", "Texas", 792727, 32.7555, -97.3308),
    ("Detroit", "Michigan", 688701, 42.3314, -83.0458),
    ("El Paso", "Texas", 674433, 31.7776, -106.4425),
    ("Memphis", "Tennessee", 653450, 35.1495, -90.0490),
    ("Seattle", "Washington", 652405, 47.6062, -122.3321),
    ("Denver", "Colorado", 649495, 39.7392, -104.9903),
    ("Washington", "District of Columbia", 646449, 38.9072, -77.0369),
    ("Boston", "Massachusetts", 645966, 42.3601, -71.0589),
    ("Nashville", "Tennessee", 634464, 36.1627, -86.7816),
    ("Baltimore", "Maryland", 622104, 39.2904, -76.6122),
    ("Oklahoma City", "Oklahoma", 610613, 35.4676, -97.5164),
    ("Louisville", "Kentucky", 609893, 38.2527, -85.7585),
    ("Portland", "Oregon", 609456, 45.5231, -122.6765),
    ("Las Vegas", "Nevada", 603488, 36.1699, -115.1398),
    ("Milwaukee", "Wisconsin", 599164, 43.0389, -87.9065),
    ("Albuquerque", "New Mexico", 556495, 35.0853, -106.6056),
    ("Tucson", "Arizona", 526116, 32.2217, -110.9265),
    ("Fresno", "California", 509924, 36.7468, -119.7726),
    ("Sacramento", "California", 479686, 38.5816, -121.4944),
    ("Long Beach", "California", 469428, 33.7701, -118.1937),
    ("Kansas City", "Missouri", 467007, 39.0997, -94.5786),
    ("Mesa", "Arizona", 457587, 33.4152, -111.8315),
    ("Virginia Beach", "Virginia", 448479, 36.8529, -75.9780),
    ("Atlanta", "Georgia", 447841, 33.7490, -84.3880),
    ("Colorado Springs", "Colorado", 439886, 38.8339, -104.8214),
    ("Omaha", "Nebraska", 434353, 41.2524, -95.9980),
    ("Raleigh", "North Carolina", 431746, 35.7796, -78.6382),
    ("Miami", "Florida", 417650, 25.7617, -80.1918),
    ("Oakland", "California", 406253, 37.8044, -122.2711),
    ("Minneapolis", "Minnesota", 400070, 44.9778, -93.2650),
    ("Tulsa", "Oklahoma", 398121, 36.1540, -95.9928),
    ("Cleveland", "Ohio", 390113, 41.4993, -81.6944),
    ("Wichita", "Kansas", 386552, 37.6889, -97.3361),
    ("Arlington", "Texas", 379577, 32.7357, -97.1081),
]


def haversine(a, b):
    """Distancia sobre la esfera terrestre (km) entre dos ciudades (lat, lon)."""
    R = 6371.0
    lat1, lon1 = math.radians(a[3]), math.radians(a[4])
    lat2, lon2 = math.radians(b[3]), math.radians(b[4])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def dist_matrix(cities):
    n = len(cities)
    return [[haversine(cities[i], cities[j]) for j in range(n)] for i in range(n)]


# ---------- 1. Programacion dinamica exacta: Held-Karp ----------
def held_karp(D):
    """
    Recursion de Bellman sobre subconjuntos visitados en vez de sobre el
    tiempo: dp[mask][j] = costo minimo de partir de la ciudad 0, visitar
    exactamente el conjunto `mask` y terminar en j. Se construye por
    tamanos de subconjunto crecientes -- el mismo principio de optimalidad
    que la induccion hacia atras, con "etapa" = |mask| en vez de t.

    O(2^n * n^2) en tiempo, O(2^n * n) en memoria -- exacto pero solo
    practico para n pequeno (~15-18 en Python puro).
    """
    n = len(D)
    FULL = (1 << n) - 1
    dp = [[math.inf] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1 << n):
        if not (mask & 1):
            continue
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            cur = dp[mask][j]
            if cur == math.inf:
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                nmask = mask | (1 << k)
                nd = cur + D[j][k]
                if nd < dp[nmask][k]:
                    dp[nmask][k] = nd
                    parent[nmask][k] = j
    best, best_j = math.inf, -1
    for j in range(1, n):
        val = dp[FULL][j] + D[j][0]
        if val < best:
            best, best_j = val, j
    tour, mask, j = [best_j], FULL, best_j
    while parent[mask][j] != -1:
        pj = parent[mask][j]
        mask ^= (1 << j)
        j = pj
        tour.append(j)
    tour.reverse()
    return best, tour


# ---------- 2. Heuristica: vecino mas cercano + 2-opt, reinicios multiples ----------
def nearest_neighbor(D, start):
    n = len(D)
    unvisited = set(range(n)) - {start}
    tour, cur = [start], start
    while unvisited:
        nxt = min(unvisited, key=lambda k: D[cur][k])
        tour.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    return tour


def tour_length(D, tour):
    n = len(tour)
    return sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))


def two_opt(D, tour):
    tour = tour[:]
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                a, b, c, d = tour[i - 1], tour[i], tour[j - 1], tour[j % n]
                if D[a][b] + D[c][d] > D[a][c] + D[b][d] + 1e-9:
                    tour[i:j] = tour[i:j][::-1]
                    improved = True
    return tour


def multi_start_heuristic(D, n_starts=None):
    n = len(D)
    starts = range(n) if n_starts is None else range(min(n_starts, n))
    best_tour, best_len = None, math.inf
    for s in starts:
        t = two_opt(D, nearest_neighbor(D, s))
        L = tour_length(D, t)
        if L < best_len:
            best_tour, best_len = t, L
    return best_len, best_tour


if __name__ == "__main__":
    n_exact = 15
    cities15 = US_CITIES[:n_exact]
    D15 = dist_matrix(cities15)

    t0 = time.time()
    opt_len, opt_tour = held_karp(D15)
    t_exact = time.time() - t0
    print(f"Held-Karp exacto, top {n_exact} ciudades: {opt_len:.1f} km  ({t_exact:.2f}s)")
    print("  Ruta:", " -> ".join(cities15[i][0] for i in opt_tour))

    nn_len = tour_length(D15, nearest_neighbor(D15, 0))
    single_len = tour_length(D15, two_opt(D15, nearest_neighbor(D15, 0)))
    multi_len, multi_tour = multi_start_heuristic(D15)
    print(f"\nComparacion en las mismas {n_exact} ciudades:")
    print(f"  vecino mas cercano (1 inicio):        {nn_len:>9.1f} km  (+{100*(nn_len/opt_len-1):.2f}% vs optimo)")
    print(f"  + 2-opt (1 inicio):                    {single_len:>9.1f} km  (+{100*(single_len/opt_len-1):.2f}% vs optimo)")
    print(f"  + 2-opt (reinicios en las {n_exact} ciudades): {multi_len:>9.1f} km  (+{100*(multi_len/opt_len-1):.2f}% vs optimo)")

    n_all = 50
    cities50 = US_CITIES[:n_all]
    D50 = dist_matrix(cities50)
    t0 = time.time()
    len50, tour50 = multi_start_heuristic(D50, n_starts=10)
    t_heur = time.time() - t0
    print(f"\nHeuristica (NN + 2-opt, 10 reinicios), top {n_all} ciudades: {len50:.1f} km  ({t_heur:.2f}s)")
    print("  (Held-Karp exacto en n=50 es inviable: 2^50 subconjuntos)")
