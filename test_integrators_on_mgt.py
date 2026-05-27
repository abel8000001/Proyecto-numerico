"""
Validación cruzada de los tres integradores contra solve_ivp (RK45) en el
modelo MGT real. Se mide el error final respecto a una solución de
referencia con tolerancias estrictas.

Este script no es parte de la biblioteca: es una verificación operativa.
"""

import numpy as np
from scipy.integrate import solve_ivp

from src.mgt_model import MGTParams, f, equilibrium
from src.integrators import INTEGRATORS


def main():
    # Régimen estable: wq suficientemente grande para garantizar convergencia
    # al equilibrio (lejos de la bifurcación de Hopf).
    params = MGTParams(wq=5e-2)
    eq = equilibrium(params)

    # Condición inicial: perturbación del 20% sobre el equilibrio para
    # excitar un transitorio observable sin alejarse a regiones con q=0.
    y0 = np.array([eq["W"] * 0.8, eq["q"] * 0.8, eq["x"] * 0.8])
    t_span = (0.0, 50.0)

    # Solución de referencia con RK45 y tolerancias estrictas.
    sol_ref = solve_ivp(
        f, t_span, y0,
        method="RK45", rtol=1e-10, atol=1e-12,
        args=(params,),
        dense_output=True,
    )
    print(f"Referencia solve_ivp: {sol_ref.nfev} evaluaciones de f, "
          f"{len(sol_ref.t)} pasos adaptativos.")
    print(f"Equilibrio analítico: W*={eq['W']:.4f}, q*={eq['q']:.4f}, "
          f"x*={eq['x']:.4f}")
    print(f"Solución de referencia en tf: "
          f"W={sol_ref.y[0, -1]:.4f}, q={sol_ref.y[1, -1]:.4f}, "
          f"x={sol_ref.y[2, -1]:.4f}")
    print()

    # Probamos cada método con un paso razonable.
    h = 1e-3
    print(f"Comparación con paso h = {h}, t_final = {t_span[1]} s")
    print("-" * 70)
    print(f"{'Método':<8}  {'err_W':>12}  {'err_q':>12}  {'err_x':>12}  "
          f"{'L_inf':>12}")
    print("-" * 70)

    for name, integrator in INTEGRATORS.items():
        t, y = integrator(f, t_span, y0, h, args=(params,))

        # Evaluar la referencia en la misma grilla temporal.
        y_ref = sol_ref.sol(t).T  # forma (n, 3)

        # Errores globales en norma L_inf por componente.
        err_W = np.max(np.abs(y[:, 0] - y_ref[:, 0]))
        err_q = np.max(np.abs(y[:, 1] - y_ref[:, 1]))
        err_x = np.max(np.abs(y[:, 2] - y_ref[:, 2]))
        err_global = max(err_W, err_q, err_x)

        print(f"{name:<8}  {err_W:>12.4e}  {err_q:>12.4e}  {err_x:>12.4e}  "
              f"{err_global:>12.4e}")

    print()
    print("Verificación adicional: convergencia al equilibrio.")
    print("-" * 70)
    h_fine = 1e-3
    t_span_long = (0.0, 150.0)  # Tiempo suficiente para converger.

    for name, integrator in INTEGRATORS.items():
        t, y = integrator(f, t_span_long, y0, h_fine, args=(params,))
        W_final, q_final, x_final = y[-1]

        err_W_eq = abs(W_final - eq["W"]) / eq["W"] * 100
        err_q_eq = abs(q_final - eq["q"]) / eq["q"] * 100

        print(f"{name:<8}  W_final={W_final:.4f} ({err_W_eq:.3f}% de W*),  "
              f"q_final={q_final:.4f} ({err_q_eq:.3f}% de q*)")


if __name__ == "__main__":
    main()
