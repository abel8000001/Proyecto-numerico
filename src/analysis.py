"""
Utilidades de análisis para el modelo MGT.

Contiene:
    - convergence_study: ejecuta cada método con varios pasos h y mide error
      global L_inf y L_2 respecto a una solución de referencia.
    - jacobian: jacobiano analítico del sistema en un punto dado.
    - linear_stability: autovalores del jacobiano en el equilibrio.
"""

from typing import Callable, Sequence
import numpy as np
from scipy.integrate import solve_ivp

from mgt_model import MGTParams, f, equilibrium, red_drop_probability
from integrators import INTEGRATORS, F_EVALS_PER_STEP


def reference_solution(
    f_rhs: Callable,
    t_span: tuple,
    y0: np.ndarray,
    args: tuple,
    rtol: float = 1e-10,
    atol: float = 1e-12,
):
    """Solución de referencia con solve_ivp (RK45, tolerancias estrictas).

    Devuelve el objeto OdeSolution con interpolación densa para evaluar
    en cualquier instante de las grillas de los métodos a probar.
    """
    sol = solve_ivp(
        f_rhs, t_span, y0,
        method="RK45", rtol=rtol, atol=atol,
        args=args, dense_output=True,
    )
    if sol.status != 0:
        raise RuntimeError(
            f"solve_ivp falló: status={sol.status}, message={sol.message}"
        )
    return sol


def convergence_study(
    f_rhs: Callable,
    t_span: tuple,
    y0: np.ndarray,
    hs: Sequence[float],
    args: tuple = (),
    component_names: Sequence[str] = ("y0", "y1", "y2"),
) -> dict:
    """Estudio de convergencia: ejecuta cada método para cada h y mide error.

    Argumentos:
        f_rhs: lado derecho del sistema.
        t_span: (t0, tf).
        y0: condición inicial.
        hs: lista de pasos de integración a probar (ordenada decreciente).
        args: argumentos adicionales para f_rhs.
        component_names: nombres de las componentes del vector de estado.

    Devuelve:
        Diccionario con estructura:
            {
                "hs": np.ndarray,
                "methods": {
                    "Euler": {
                        "err_inf_per_component": np.ndarray (n_h, d),
                        "err_inf_global":        np.ndarray (n_h,),
                        "err_l2_global":         np.ndarray (n_h,),
                        "n_feval":               np.ndarray (n_h,),
                        "slope_inf":             float,  # pendiente empírica
                    },
                    "RK4":  {...},
                    "ABM4": {...},
                },
                "component_names": tuple,
                "t_span": tuple,
            }
    """
    hs = np.asarray(hs, dtype=float)
    sol_ref = reference_solution(f_rhs, t_span, y0, args=args)

    results = {"hs": hs, "methods": {}, "component_names": tuple(component_names),
               "t_span": t_span}

    d = len(y0)

    for name, integrator in INTEGRATORS.items():
        err_inf_per_comp = np.zeros((len(hs), d))
        err_inf_global = np.zeros(len(hs))
        err_l2_global = np.zeros(len(hs))
        n_feval = np.zeros(len(hs), dtype=int)

        for i, h in enumerate(hs):
            t, y = integrator(f_rhs, t_span, y0, h, args=args)
            y_ref = sol_ref.sol(t).T  # (n_t, d)

            diff = y - y_ref
            err_inf_per_comp[i, :] = np.max(np.abs(diff), axis=0)
            err_inf_global[i] = np.max(err_inf_per_comp[i, :])
            # L2 global: norma RMS sobre tiempo y componentes.
            err_l2_global[i] = np.sqrt(np.mean(diff**2))

            # Costo: número de pasos por evaluaciones de f por paso.
            n_feval[i] = (len(t) - 1) * F_EVALS_PER_STEP[name]

        # Pendiente empírica (regresión lineal en log-log).
        slope_inf = np.polyfit(np.log(hs), np.log(err_inf_global), 1)[0]

        results["methods"][name] = {
            "err_inf_per_component": err_inf_per_comp,
            "err_inf_global": err_inf_global,
            "err_l2_global": err_l2_global,
            "n_feval": n_feval,
            "slope_inf": slope_inf,
        }

    return results


def jacobian(y: np.ndarray, params: MGTParams) -> np.ndarray:
    """Jacobiano analítico del sistema MGT (versión sin retardo) en y.

    Sistema:
        dW/dt = 1/R - W^2/(2R) * p(x)
        dq/dt = -C + N*W/R
        dx/dt = wq*C*(q - x)

    El jacobiano es:
        J = | -W*p(x)/R       0              -W^2/(2R) * p'(x) |
            |  N/R            0               0                |
            |  0              wq*C           -wq*C             |
    """
    W, q, x = y

    # p y p'(x) (derivada de la curva RED).
    p_val = red_drop_probability(x, params)

    if params.q_min < x < params.q_max:
        dp_dx = params.p_max / (params.q_max - params.q_min)
    else:
        dp_dx = 0.0  # En las zonas saturadas p es constante.

    J = np.array([
        [-W * p_val / params.R, 0.0, -W * W / (2.0 * params.R) * dp_dx],
        [params.N / params.R,   0.0, 0.0],
        [0.0,                   params.wq * params.C, -params.wq * params.C],
    ])
    return J


def linear_stability(params: MGTParams) -> dict:
    """Análisis de estabilidad lineal en el equilibrio.

    Devuelve los autovalores del jacobiano evaluado en (W*, q*, x*) y
    metadatos útiles (parte real máxima, frecuencia oscilatoria si aplica).
    """
    eq = equilibrium(params)
    y_eq = np.array([eq["W"], eq["q"], eq["x"]])
    J = jacobian(y_eq, params)
    eigs = np.linalg.eigvals(J)

    re_max = np.max(eigs.real)
    # Frecuencia oscilatoria del par complejo dominante (si existe).
    complex_eigs = eigs[np.abs(eigs.imag) > 1e-12]
    if len(complex_eigs) > 0:
        # Tomamos el de parte real máxima (dominante).
        idx = np.argmax(complex_eigs.real)
        omega = abs(complex_eigs[idx].imag)
    else:
        omega = None

    return {
        "eigenvalues": eigs,
        "re_max": re_max,
        "omega_dominant": omega,
        "stable": bool(re_max < 0),
        "jacobian": J,
        "equilibrium": eq,
    }
