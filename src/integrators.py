"""
Integradores numéricos de paso fijo para sistemas de EDOs de primer orden.

Implementa tres métodos programados desde cero:
    - Euler explícito (orden 1)
    - Runge-Kutta clásico de orden 4 (RK4)
    - Adams-Bashforth-Moulton 4 predictor-corrector (PECE, orden 4)

Interfaz común:
    t, y = integrator(f, t_span, y0, h, args=...)

donde:
    f       : función f(t, y, *args) que devuelve dy/dt como np.ndarray.
    t_span  : tupla (t0, t_final).
    y0      : condición inicial, np.ndarray de tamaño d.
    h       : paso de integración (constante).
    args    : tupla de argumentos extra pasados a f.

Devuelven:
    t : np.ndarray de tiempos, tamaño n+1.
    y : np.ndarray de soluciones, tamaño (n+1, d).
"""

from typing import Callable, Tuple
import numpy as np


def _build_time_grid(t_span: Tuple[float, float], h: float) -> np.ndarray:
    """Construye la grilla temporal uniforme de paso h sobre [t0, t_final].

    Si (t_final - t0) no es múltiplo exacto de h, el último paso se ajusta
    para terminar exactamente en t_final.
    """
    t0, tf = t_span
    n = int(np.ceil((tf - t0) / h))
    t = t0 + h * np.arange(n + 1)
    t[-1] = tf  # Forzar terminación exacta para evitar deriva por redondeo.
    return t


def euler(
    f: Callable,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    h: float,
    args: tuple = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """Método de Euler explícito (orden 1).

    Esquema:
        y_{n+1} = y_n + h * f(t_n, y_n)

    Una evaluación de f por paso.
    """
    t = _build_time_grid(t_span, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for n in range(len(t) - 1):
        dt = t[n + 1] - t[n]
        y[n + 1] = y[n] + dt * f(t[n], y[n], *args)

    return t, y


def rk4(
    f: Callable,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    h: float,
    args: tuple = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """Runge-Kutta clásico de cuarto orden (orden 4).

    Esquema:
        k1 = f(t_n,         y_n)
        k2 = f(t_n + h/2,   y_n + h/2 * k1)
        k3 = f(t_n + h/2,   y_n + h/2 * k2)
        k4 = f(t_n + h,     y_n + h   * k3)
        y_{n+1} = y_n + h/6 * (k1 + 2 k2 + 2 k3 + k4)

    Cuatro evaluaciones de f por paso.
    """
    t = _build_time_grid(t_span, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for n in range(len(t) - 1):
        dt = t[n + 1] - t[n]
        k1 = f(t[n], y[n], *args)
        k2 = f(t[n] + dt / 2, y[n] + dt / 2 * k1, *args)
        k3 = f(t[n] + dt / 2, y[n] + dt / 2 * k2, *args)
        k4 = f(t[n] + dt, y[n] + dt * k3, *args)
        y[n + 1] = y[n] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y


def abm4(
    f: Callable,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    h: float,
    args: tuple = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """Adams-Bashforth-Moulton 4 (predictor-corrector PECE, orden 4).

    Predictor (Adams-Bashforth 4, explícito):
        y_p = y_n + h/24 * (55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})

    Corrector (Adams-Moulton 4, implícito evaluado una vez):
        y_{n+1} = y_n + h/24 * (9 f_p + 19 f_n - 5 f_{n-1} + f_{n-2})

    donde f_p = f(t_{n+1}, y_p). Modo PECE: una iteración del corrector.

    Arranque: los primeros tres pasos se calculan con RK4 para acumular
    el historial de evaluaciones de f necesario.

    Evaluaciones de f por paso (en régimen): 2 (una para el predictor,
    una para el corrector). El método es así más eficiente que RK4
    para horizontes largos a igual orden de precisión.
    """
    t = _build_time_grid(t_span, h)
    d = len(y0)
    y = np.zeros((len(t), d))
    y[0] = y0

    # ---- Arranque con RK4 para los primeros 3 pasos ----
    # Necesitamos f en n=0,1,2,3 para iniciar el multipaso en n=3.
    for n in range(min(3, len(t) - 1)):
        dt = t[n + 1] - t[n]
        k1 = f(t[n], y[n], *args)
        k2 = f(t[n] + dt / 2, y[n] + dt / 2 * k1, *args)
        k3 = f(t[n] + dt / 2, y[n] + dt / 2 * k2, *args)
        k4 = f(t[n] + dt, y[n] + dt * k3, *args)
        y[n + 1] = y[n] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    if len(t) <= 4:
        return t, y  # Horizonte tan corto que sólo hay arranque.

    # Cache de evaluaciones de f en los puntos ya conocidos.
    f_hist = [f(t[i], y[i], *args) for i in range(4)]
    # f_hist[k] corresponde a t[k], y[k] inicialmente; se irá rotando.

    # ---- Bucle principal: de n=3 en adelante ----
    for n in range(3, len(t) - 1):
        dt = t[n + 1] - t[n]
        f_n, f_nm1, f_nm2, f_nm3 = f_hist[3], f_hist[2], f_hist[1], f_hist[0]

        # Predictor AB4 (explícito).
        y_pred = y[n] + dt / 24 * (
            55 * f_n - 59 * f_nm1 + 37 * f_nm2 - 9 * f_nm3
        )

        # Evaluación de f en el punto predicho.
        f_pred = f(t[n + 1], y_pred, *args)

        # Corrector AM4 (implícito, evaluado una vez).
        y[n + 1] = y[n] + dt / 24 * (
            9 * f_pred + 19 * f_n - 5 * f_nm1 + f_nm2
        )

        # Actualizar historial: descartamos f_{n-3}, agregamos f_{n+1}.
        # Reevaluamos f en el punto corregido para mejor precisión.
        f_hist = [f_hist[1], f_hist[2], f_hist[3], f(t[n + 1], y[n + 1], *args)]

    return t, y


# Diccionario para iterar fácilmente sobre los métodos en los tests.
INTEGRATORS = {
    "Euler": euler,
    "RK4": rk4,
    "ABM4": abm4,
}


# Evaluaciones de f por paso (en régimen, despreciando arranque para ABM4).
F_EVALS_PER_STEP = {
    "Euler": 1,
    "RK4": 4,
    "ABM4": 2,
}


# -----------------------------------------------------------------------------
# Verificación contra el oscilador armónico simple.
#
# Sistema: y'' + omega^2 y = 0  =>  como primer orden,
#   dy1/dt = y2
#   dy2/dt = -omega^2 y1
# Solución analítica con y1(0)=A, y2(0)=0:
#   y1(t) = A cos(omega t),   y2(t) = -A omega sin(omega t)
#
# Lo usamos porque tiene solución cerrada y los errores de cada método
# deberían caer con las pendientes teóricas 1 (Euler) y 4 (RK4, ABM4).
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Verificación de integradores: oscilador armónico")
    print("=" * 60)

    omega = 2.0 * np.pi
    A = 1.0

    def f_harmonic(t, y, omega):
        return np.array([y[1], -omega * omega * y[0]])

    def y_exact(t, omega, A):
        return np.array([A * np.cos(omega * t), -A * omega * np.sin(omega * t)])

    t0, tf = 0.0, 1.0  # Un período completo.
    y0 = np.array([A, 0.0])

    # Pasos sucesivamente más finos.
    hs = [1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4]

    expected_orders = {"Euler": 1, "RK4": 4, "ABM4": 4}

    for name, integrator in INTEGRATORS.items():
        errs = []
        for h in hs:
            t, y = integrator(f_harmonic, (t0, tf), y0, h, args=(omega,))
            err = np.max(np.abs(y[-1] - y_exact(tf, omega, A)))
            errs.append(err)

        # Pendiente empírica entre el penúltimo y último h.
        slope = np.log(errs[-2] / errs[-1]) / np.log(hs[-2] / hs[-1])

        print(f"\n{name}  (orden teórico = {expected_orders[name]})")
        print(f"  {'h':>10}  {'err_inf':>12}")
        for h, err in zip(hs, errs):
            print(f"  {h:>10.2e}  {err:>12.4e}")
        print(f"  Pendiente empírica final: {slope:.3f}")
