"""
Modelo fluido TCP/RED de Misra-Gomez-Towsley (MGT).

Implementa el sistema de tres EDOs no lineales acopladas que describen:
- W(t): ventana de congestión TCP promedio (paquetes)
- q(t): tamaño instantáneo de la cola en el router (paquetes)
- x(t): cola promedio EWMA usada por RED (paquetes)

Se trabaja con la aproximación sin retardo (R aproximadamente constante).

Referencias:
    Misra, V., Gong, W.-B., Towsley, D. (2000). Fluid-based analysis of a
    network of AQM routers supporting TCP flows with an application to RED.
    ACM SIGCOMM Computer Communication Review, 30(4), 151-160.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MGTParams:
    """Parámetros del modelo MGT-RED.

    Valores nominales tomados de Hollot, Misra, Towsley, Gong (2002),
    correspondientes a un enlace de 15 Mbps con paquetes de 500 B.

    Atributos:
        N: número de flujos TCP activos.
        C: capacidad del enlace en paquetes/s.
        R: round-trip time efectivo en segundos.
        q_min: umbral mínimo de RED en paquetes.
        q_max: umbral máximo de RED en paquetes.
        p_max: probabilidad máxima de descarte en la zona lineal de RED.
        wq: peso del filtro EWMA de RED.
        B: tamaño físico del buffer del router en paquetes.
    """
    N: float = 60.0
    C: float = 3750.0
    R: float = 0.246
    q_min: float = 150.0
    q_max: float = 700.0
    p_max: float = 0.1
    wq: float = 1.33e-5
    B: float = 800.0


def red_drop_probability(x: float, params: MGTParams) -> float:
    """Probabilidad de descarte RED en función de la cola promedio x.

    La curva es lineal por tramos:
    - p = 0 si x <= q_min
    - p crece linealmente entre q_min y q_max hasta p_max
    - p = p_max si x >= q_max (se omite la zona de drop forzado a 1)

    Argumentos:
        x: cola promedio EWMA en paquetes.
        params: parámetros del modelo.

    Devuelve:
        Probabilidad de descarte en [0, p_max].
    """
    if x <= params.q_min:
        return 0.0
    if x >= params.q_max:
        return params.p_max
    return (x - params.q_min) / (params.q_max - params.q_min) * params.p_max


def f(t: float, y: np.ndarray, params: MGTParams) -> np.ndarray:
    """Lado derecho del sistema MGT en forma de primer orden.

    Sistema:
        dW/dt = 1/R - (W^2 / (2 R)) * p(x)
        dq/dt = -C + N * W / R          (saturada a q >= 0)
        dx/dt = wq * C * (q - x)

    Argumentos:
        t: tiempo (no aparece explícitamente porque el sistema es autónomo).
        y: vector de estado [W, q, x].
        params: parámetros del modelo.

    Devuelve:
        Vector [dW/dt, dq/dt, dx/dt] como np.ndarray de tamaño 3.
    """
    W, q, x = y
    p = red_drop_probability(x, params)

    dW = 1.0 / params.R - (W * W) / (2.0 * params.R) * p
    dq = -params.C + params.N * W / params.R

    # Saturación física: la cola no puede ser negativa.
    if q <= 0.0 and dq < 0.0:
        dq = 0.0

    dx = params.wq * params.C * (q - x)

    return np.array([dW, dq, dx])


def equilibrium(params: MGTParams) -> dict:
    """Calcula el punto de equilibrio analítico del sistema MGT.

    Imponiendo dW/dt = dq/dt = dx/dt = 0 se obtienen las expresiones cerradas:
        W* = R * C / N
        p* = 2 * N^2 / (R^2 * C^2)
        x* = q* = q_min + (p* / p_max) * (q_max - q_min)

    Argumentos:
        params: parámetros del modelo.

    Devuelve:
        Diccionario con claves 'W', 'q', 'x', 'p'.

    Lanza:
        ValueError si p* cae fuera de la zona lineal de RED (modelo
        no válido en ese régimen, requeriría saturación).
    """
    W_eq = params.R * params.C / params.N
    p_eq = 2.0 * params.N**2 / (params.R**2 * params.C**2)

    if p_eq <= 0 or p_eq >= params.p_max:
        raise ValueError(
            f"p* = {p_eq:.4f} fuera de la zona lineal de RED [0, {params.p_max}]. "
            "Los parámetros eligen un equilibrio en saturación; revisar N, C o R."
        )

    q_eq = params.q_min + (p_eq / params.p_max) * (params.q_max - params.q_min)
    x_eq = q_eq  # El EWMA en equilibrio iguala a la cola instantánea.

    return {"W": W_eq, "q": q_eq, "x": x_eq, "p": p_eq}


# -----------------------------------------------------------------------------
# Verificación: si se ejecuta este módulo directamente, se calcula el
# equilibrio con los parámetros nominales y se sustituye en f para confirmar
# que el residual es numéricamente cero.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    params = MGTParams()
    eq = equilibrium(params)

    print("=" * 60)
    print("Modelo MGT: Verificación del equilibrio analítico")
    print("=" * 60)
    print(f"Parámetros nominales: N={params.N}, C={params.C}, R={params.R}")
    print(f"                      wq={params.wq}, p_max={params.p_max}")
    print(f"                      q_min={params.q_min}, q_max={params.q_max}")
    print()
    print(f"Equilibrio analítico:")
    print(f"  W* = {eq['W']:.6f} pkt")
    print(f"  q* = {eq['q']:.6f} pkt")
    print(f"  x* = {eq['x']:.6f} pkt")
    print(f"  p* = {eq['p']:.6e}")
    print()

    # Sustituir el equilibrio en f y verificar que devuelve ~0.
    y_eq = np.array([eq["W"], eq["q"], eq["x"]])
    residual = f(0.0, y_eq, params)

    print(f"Residual f(y*) = {residual}")
    print(f"  ||residual||_inf = {np.max(np.abs(residual)):.2e}")
    print()

    tol = 1e-10
    if np.max(np.abs(residual)) < tol:
        print(f"OK: el equilibrio satisface f(y*) = 0 (tol = {tol:.0e}).")
    else:
        print("ERROR: el equilibrio no anula f; revisar derivaciones.")
