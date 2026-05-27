"""
Extensión del modelo MGT a dos rutas paralelas con balanceo dinámico.

Cada ruta k = 1, 2 tiene su propio cuello de botella (C_k, R_k) y AQM con
filtro RED. Una fracción phi(t) en [0, 1] del tráfico va por la ruta 1, el
resto por la ruta 2. La dinámica de splitting es un descenso de gradiente
sobre la diferencia de probabilidades de descarte (consistente con el
principio de Wardrop: en equilibrio las pérdidas se igualan).

Vector de estado (7 componentes):
    y = [W1, q1, x1, W2, q2, x2, phi]
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

from mgt_model import MGTParams, red_drop_probability


@dataclass
class TwoRouteParams:
    """Parámetros del sistema de dos rutas.

    Atributos:
        N: número total de flujos.
        route1, route2: parámetros MGT de cada ruta (capacidad, RTT, RED).
        alpha: ganancia del balanceador (rapidez de ajuste de phi).
    """
    N: float = 60.0
    route1: MGTParams = field(default_factory=lambda: MGTParams(
        wq=5e-2, C=3750.0, R=0.246))
    route2: MGTParams = field(default_factory=lambda: MGTParams(
        wq=5e-2, C=2500.0, R=0.246))
    alpha: float = 5.0  # Ganancia del balanceador.

    def __post_init__(self):
        # Nota: cada route_k mantiene su N en su propio MGTParams, pero
        # en el sistema dinámico el N efectivo es N_k = N * phi (o 1-phi).
        # Por eso usamos el atributo N de este contenedor, no el de los
        # MGTParams individuales (que se ignora dentro de f_two_routes).
        pass


def f_two_routes(t: float, y: np.ndarray, params: TwoRouteParams) -> np.ndarray:
    """Lado derecho del sistema MGT de dos rutas (7 EDOs).

    Argumentos:
        t: tiempo (sistema autónomo, no aparece).
        y: vector [W1, q1, x1, W2, q2, x2, phi].
        params: parámetros del sistema de dos rutas.

    Devuelve:
        Vector dy/dt de tamaño 7 como np.ndarray.
    """
    W1, q1, x1, W2, q2, x2, phi = y
    r1, r2 = params.route1, params.route2

    # Saturación de phi a [0, 1] para evitar fracciones negativas o > 1.
    phi_eff = max(0.0, min(1.0, phi))
    N1 = params.N * phi_eff
    N2 = params.N * (1.0 - phi_eff)

    # Probabilidades de descarte en cada ruta.
    p1 = red_drop_probability(x1, r1)
    p2 = red_drop_probability(x2, r2)

    # Dinámica TCP/AQM en cada ruta.
    dW1 = 1.0 / r1.R - (W1 * W1) / (2.0 * r1.R) * p1
    dq1 = -r1.C + N1 * W1 / r1.R
    if q1 <= 0.0 and dq1 < 0.0:
        dq1 = 0.0
    dx1 = r1.wq * r1.C * (q1 - x1)

    dW2 = 1.0 / r2.R - (W2 * W2) / (2.0 * r2.R) * p2
    dq2 = -r2.C + N2 * W2 / r2.R
    if q2 <= 0.0 and dq2 < 0.0:
        dq2 = 0.0
    dx2 = r2.wq * r2.C * (q2 - x2)

    # Balanceo: descenso de gradiente en la diferencia de pérdidas.
    # En equilibrio dphi/dt = 0 ==> p1 = p2 (principio de Wardrop).
    dphi = -params.alpha * (p1 - p2)
    # Saturación: phi no puede salir de [0, 1].
    if (phi >= 1.0 and dphi > 0.0) or (phi <= 0.0 and dphi < 0.0):
        dphi = 0.0

    return np.array([dW1, dq1, dx1, dW2, dq2, dx2, dphi])


def equilibrium_two_routes(params: TwoRouteParams) -> dict:
    """Equilibrio analítico del sistema de dos rutas.

    En equilibrio (Wardrop): p1* = p2* = p*. Combinado con las relaciones
    de cada ruta dW_k/dt = 0 y dq_k/dt = 0:

        W_k* = R_k * C_k / N_k         (con N_k = N * phi*  o  N * (1-phi*))
        p_k* = 2 / W_k*^2 = 2 * N_k^2 / (R_k^2 * C_k^2)

    Imponiendo p1* = p2*:
        N_1 / (R_1 * C_1) = N_2 / (R_2 * C_2)
        N * phi* / (R_1 C_1) = N (1-phi*) / (R_2 C_2)
        phi* = (R_1 C_1) / (R_1 C_1 + R_2 C_2)

    Observación: cuando R_1 = R_2, esto se reduce a
        phi* = C_1 / (C_1 + C_2)
    es decir, la fracción de tráfico es proporcional a la capacidad de la ruta.

    Devuelve:
        Diccionario con W_k, q_k, x_k, p_k para k=1,2, y phi*.
    """
    r1, r2 = params.route1, params.route2
    N = params.N

    # Splitting de Wardrop.
    phi_eq = (r1.R * r1.C) / (r1.R * r1.C + r2.R * r2.C)

    N1 = N * phi_eq
    N2 = N * (1.0 - phi_eq)

    W1_eq = r1.R * r1.C / N1
    W2_eq = r2.R * r2.C / N2
    p_eq = 2.0 / W1_eq**2  # = 2 / W2_eq^2 también (Wardrop).

    # Validar que ambos p estén en la zona lineal de RED.
    for k, r, p in [(1, r1, p_eq), (2, r2, p_eq)]:
        if not (0 < p < r.p_max):
            raise ValueError(
                f"p_{k}* = {p:.4f} fuera de zona lineal de RED [0, {r.p_max}]"
            )

    q1_eq = r1.q_min + (p_eq / r1.p_max) * (r1.q_max - r1.q_min)
    q2_eq = r2.q_min + (p_eq / r2.p_max) * (r2.q_max - r2.q_min)

    return {
        "W1": W1_eq, "q1": q1_eq, "x1": q1_eq, "p1": p_eq,
        "W2": W2_eq, "q2": q2_eq, "x2": q2_eq, "p2": p_eq,
        "phi": phi_eq,
        "N1": N1, "N2": N2,
    }


# -----------------------------------------------------------------------------
# Verificación: equilibrio analítico vs. residual de f.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Caso asimétrico: ruta 1 más rápida (C1=3750) que ruta 2 (C2=2500).
    params = TwoRouteParams()
    eq = equilibrium_two_routes(params)

    print("=" * 65)
    print("Modelo de dos rutas - Verificación del equilibrio Wardrop")
    print("=" * 65)
    print(f"N total = {params.N}")
    print(f"Ruta 1: C1={params.route1.C} pkt/s, R1={params.route1.R} s")
    print(f"Ruta 2: C2={params.route2.C} pkt/s, R2={params.route2.R} s")
    print(f"alpha (ganancia balanceador) = {params.alpha}")
    print()
    print(f"Equilibrio analítico:")
    print(f"  phi*  = {eq['phi']:.6f}  (esperado: C1/(C1+C2) = "
          f"{params.route1.C/(params.route1.C + params.route2.C):.6f})")
    print(f"  N1*   = {eq['N1']:.4f} flujos por ruta 1")
    print(f"  N2*   = {eq['N2']:.4f} flujos por ruta 2")
    print(f"  W1*   = {eq['W1']:.6f},  W2* = {eq['W2']:.6f}")
    print(f"  q1*   = {eq['q1']:.4f},  q2* = {eq['q2']:.4f}")
    print(f"  p1*=p2* = {eq['p1']:.6e}  (Wardrop: pérdidas iguales)")
    print()

    # Verificación numérica: f en el equilibrio debe dar ~0.
    y_eq = np.array([eq["W1"], eq["q1"], eq["x1"],
                     eq["W2"], eq["q2"], eq["x2"],
                     eq["phi"]])
    residual = f_two_routes(0.0, y_eq, params)
    print(f"Residual f(y*) = {residual}")
    print(f"  ||residual||_inf = {np.max(np.abs(residual)):.2e}")
    if np.max(np.abs(residual)) < 1e-10:
        print("OK: el equilibrio satisface f(y*) = 0.")
    else:
        print("ERROR: el equilibrio no anula f; revisar derivaciones.")
