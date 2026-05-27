"""
Genera la Tabla 2 (texto + LaTeX) y la Figura 1 del informe.

Tabla 2: convergencia empírica de los tres métodos en el modelo MGT.
         Columnas: h, err_inf en W/q/x, pendiente empírica, n_feval.

Figura 1: (a) log-log error vs h para los tres métodos, con pendientes
              teóricas como referencia.
          (b) error vs n_feval (eficiencia).

Adicionalmente: tabla pequeña ilustrando el límite de estabilidad
(qué pasa con Euler en h grande).

Uso:
    python -m src.plots.convergence
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from mgt_model import MGTParams, f, equilibrium
from analysis import convergence_study, linear_stability


# Directorio de figuras (relativo a la raíz del repo).
FIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)


def run_main_study():
    """Estudio principal de convergencia en régimen estable.

    Se usa una perturbación pequeña (5%) sobre el equilibrio para mantener
    la trayectoria estrictamente dentro de la zona lineal de RED durante
    todo el transitorio. Esto evita que los cruces de los umbrales q_min,
    q_max introduzcan no diferenciabilidades en p(x), las cuales degradan
    el orden empírico de los métodos de alto orden.
    """
    # Régimen estable lejos de la bifurcación.
    params = MGTParams(wq=5e-2)
    eq = equilibrium(params)

    # Perturbación inicial del 5%: trayectoria permanece en zona lineal RED.
    y0 = np.array([eq["W"] * 0.95, eq["q"] * 0.95, eq["x"] * 0.95])
    t_span = (0.0, 30.0)

    # Pasos en el régimen estable de los tres métodos.
    hs = [5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4]

    res = convergence_study(
        f, t_span, y0, hs, args=(params,),
        component_names=("W", "q", "x"),
    )
    return res, params, eq, y0, t_span


def run_stability_demo():
    """Demostración del límite de estabilidad: Euler con h grande diverge."""
    params = MGTParams(wq=5e-2)
    eq = equilibrium(params)
    y0 = np.array([eq["W"] * 0.95, eq["q"] * 0.95, eq["x"] * 0.95])
    t_span = (0.0, 30.0)

    # Pasos que cruzan el límite de estabilidad de Euler (h_crit ~ 0.011 s).
    hs = [5e-2, 1e-2, 5e-3]

    res = convergence_study(
        f, t_span, y0, hs, args=(params,),
        component_names=("W", "q", "x"),
    )
    return res, hs


def print_text_table(res):
    """Imprime la Tabla 2 en formato texto plano."""
    hs = res["hs"]
    print("=" * 80)
    print("Tabla 2 - Convergencia empírica (régimen estable, t_span=[0,30])")
    print("=" * 80)
    print(f"{'Método':<8}  {'h':>8}  {'err W':>10}  {'err q':>10}  "
          f"{'err x':>10}  {'L_inf':>10}  {'n_feval':>9}")
    print("-" * 80)

    for name in ["Euler", "RK4", "ABM4"]:
        m = res["methods"][name]
        for i, h in enumerate(hs):
            err_comp = m["err_inf_per_component"][i]
            print(f"{name:<8}  {h:>8.0e}  {err_comp[0]:>10.3e}  "
                  f"{err_comp[1]:>10.3e}  {err_comp[2]:>10.3e}  "
                  f"{m['err_inf_global'][i]:>10.3e}  {m['n_feval'][i]:>9d}")
        print(f"{'':>8}  pendiente empírica (log-log, err_inf vs h): "
              f"{m['slope_inf']:.3f}")
        print("-" * 80)


def print_latex_table(res):
    """Imprime la Tabla 2 en formato LaTeX (tabular)."""
    hs = res["hs"]
    print("\n--- LaTeX (Tabla 2) ---")
    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\hline")
    print(r"Método & $h$ & $\|e_W\|_\infty$ & $\|e_q\|_\infty$ & "
          r"$\|e_x\|_\infty$ & $\|e\|_\infty$ & $N_f$ \\")
    print(r"\hline")
    for name in ["Euler", "RK4", "ABM4"]:
        m = res["methods"][name]
        for i, h in enumerate(hs):
            err_comp = m["err_inf_per_component"][i]
            print(f"{name} & {h:.0e} & {err_comp[0]:.2e} & "
                  f"{err_comp[1]:.2e} & {err_comp[2]:.2e} & "
                  f"{m['err_inf_global'][i]:.2e} & {m['n_feval'][i]} \\\\")
        print(rf"\multicolumn{{6}}{{r}}{{Pendiente empírica: "
              rf"{m['slope_inf']:.3f}}} \\")
        print(r"\hline")
    print(r"\end{tabular}")


def make_figure_1(res, save_path):
    """Figura 1: convergencia y eficiencia."""
    hs = res["hs"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = {"Euler": "#1f77b4", "RK4": "#d62728", "ABM4": "#2ca02c"}
    markers = {"Euler": "o", "RK4": "s", "ABM4": "^"}

    # (a) Error L_inf vs h en log-log.
    ax = axes[0]
    for name in ["Euler", "RK4", "ABM4"]:
        m = res["methods"][name]
        ax.loglog(hs, m["err_inf_global"], marker=markers[name],
                  color=colors[name], label=f"{name} (pend.={m['slope_inf']:.2f})",
                  linewidth=1.6, markersize=7)

    # Líneas teóricas de referencia O(h) y O(h^4).
    h_ref = np.array([hs[-1], hs[0]])
    err_h = res["methods"]["Euler"]["err_inf_global"][0] * (h_ref / hs[0])**1
    err_h4 = res["methods"]["RK4"]["err_inf_global"][0] * (h_ref / hs[0])**4
    ax.loglog(h_ref, err_h, "--", color="gray", alpha=0.6,
              linewidth=1, label=r"$\mathcal{O}(h)$")
    ax.loglog(h_ref, err_h4, ":", color="gray", alpha=0.6,
              linewidth=1, label=r"$\mathcal{O}(h^4)$")

    ax.set_xlabel(r"Paso $h$ [s]")
    ax.set_ylabel(r"Error global $\|\mathbf{e}\|_\infty$")
    ax.set_title("(a) Convergencia")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # (b) Error vs n_feval (eficiencia).
    ax = axes[1]
    for name in ["Euler", "RK4", "ABM4"]:
        m = res["methods"][name]
        ax.loglog(m["n_feval"], m["err_inf_global"], marker=markers[name],
                  color=colors[name], label=name,
                  linewidth=1.6, markersize=7)

    ax.set_xlabel(r"Número de evaluaciones de $f$")
    ax.set_ylabel(r"Error global $\|\mathbf{e}\|_\infty$")
    ax.set_title("(b) Eficiencia")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Fig. 1 - Convergencia y eficiencia de los métodos numéricos "
                 "en el modelo MGT", fontsize=11, y=1.01)
    fig.tight_layout()

    fig.savefig(save_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en {save_path}.{{png,pdf}}")


def print_stability_demo(res_demo, hs_demo):
    """Imprime la pequeña tabla del límite de estabilidad."""
    print()
    print("=" * 80)
    print("Demostración del límite de estabilidad (h grande)")
    print("=" * 80)
    print(f"Para wq=5e-2, autovalor rápido lambda~-187.6, límite Euler "
          f"h_crit ~ 2/|lambda| = {2/187.6:.4f} s")
    print()
    print(f"{'Método':<8}  {'h':>8}  {'L_inf':>12}  {'estado':<20}")
    print("-" * 80)
    for name in ["Euler", "RK4", "ABM4"]:
        m = res_demo["methods"][name]
        for i, h in enumerate(hs_demo):
            err = m["err_inf_global"][i]
            if np.isnan(err) or err > 1e6:
                state = "DIVERGE"
                err_str = "  inestable"
            else:
                state = "estable"
                err_str = f"{err:.3e}"
            print(f"{name:<8}  {h:>8.0e}  {err_str:>12}  {state:<20}")


def make_figure_2(save_path):
    """Figura 2: régimen estable. W(t), q(t), plano de fases (W, q).

    Simulación con RK4 a paso h=1e-3, partiendo de una perturbación del 5%
    sobre el equilibrio. Se anotan los valores asintóticos analíticos.
    """
    from integrators import rk4

    params = MGTParams(wq=5e-2)
    eq = equilibrium(params)
    y0 = np.array([eq["W"] * 0.95, eq["q"] * 0.95, eq["x"] * 0.95])
    t_span = (0.0, 30.0)
    h = 1e-3

    t, y = rk4(f, t_span, y0, h, args=(params,))
    W, q, x = y[:, 0], y[:, 1], y[:, 2]

    # Verificación cuantitativa: error relativo al equilibrio en t_final.
    err_W = abs(W[-1] - eq["W"]) / eq["W"] * 100
    err_q = abs(q[-1] - eq["q"]) / eq["q"] * 100
    print(f"\nFigura 2 - Convergencia al equilibrio (t={t_span[1]} s):")
    print(f"  W_final = {W[-1]:.6f}  (W* = {eq['W']:.6f}, "
          f"err rel = {err_W:.4f}%)")
    print(f"  q_final = {q[-1]:.6f}  (q* = {eq['q']:.6f}, "
          f"err rel = {err_q:.4f}%)")

    # Layout: 2 paneles arriba (W, q), 1 grande abajo (plano de fases).
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.35, wspace=0.25)

    # (a) W(t).
    ax_W = fig.add_subplot(gs[0, 0])
    ax_W.plot(t, W, color="#1f77b4", linewidth=1.3)
    ax_W.axhline(eq["W"], color="gray", linestyle="--", linewidth=0.9,
                 label=fr"$W^* = {eq['W']:.3f}$")
    ax_W.set_xlabel(r"Tiempo $t$ [s]")
    ax_W.set_ylabel(r"Ventana TCP $W$ [pkt]")
    ax_W.set_title("(a) Ventana de congestión")
    ax_W.legend(fontsize=9, loc="lower right")
    ax_W.grid(True, alpha=0.3)

    # (b) q(t).
    ax_q = fig.add_subplot(gs[0, 1])
    ax_q.plot(t, q, color="#d62728", linewidth=1.3)
    ax_q.axhline(eq["q"], color="gray", linestyle="--", linewidth=0.9,
                 label=fr"$q^* = {eq['q']:.2f}$")
    ax_q.set_xlabel(r"Tiempo $t$ [s]")
    ax_q.set_ylabel(r"Cola $q$ [pkt]")
    ax_q.set_title("(b) Tamaño instantáneo de cola")
    ax_q.legend(fontsize=9, loc="lower right")
    ax_q.grid(True, alpha=0.3)

    # (c) Plano de fases (W, q) - fila completa abajo.
    ax_phase = fig.add_subplot(gs[1, :])
    # Coloreamos por tiempo para visualizar la dirección de la trayectoria.
    points = np.array([W, q]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap="viridis",
                        norm=plt.Normalize(t.min(), t.max()),
                        linewidth=1.6)
    lc.set_array(t[:-1])
    ax_phase.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax_phase, pad=0.02)
    cbar.set_label(r"Tiempo $t$ [s]", fontsize=9)

    # Marcar CI y equilibrio.
    ax_phase.plot(y0[0], y0[1], "o", color="black", markersize=8,
                  label=fr"CI $(W_0, q_0) = ({y0[0]:.2f}, {y0[1]:.2f})$")
    ax_phase.plot(eq["W"], eq["q"], "*", color="red", markersize=14,
                  label=fr"Equilibrio $(W^*, q^*)$", zorder=5)

    ax_phase.set_xlim(W.min() - 0.1, W.max() + 0.1)
    ax_phase.set_ylim(q.min() - 5, q.max() + 5)
    ax_phase.set_xlabel(r"Ventana TCP $W$ [pkt]")
    ax_phase.set_ylabel(r"Cola $q$ [pkt]")
    ax_phase.set_title("(c) Plano de fases - espiral logarítmica hacia el equilibrio")
    ax_phase.legend(fontsize=9, loc="best")
    ax_phase.grid(True, alpha=0.3)

    fig.suptitle("Fig. 2 - Régimen estable: convergencia al equilibrio "
                 fr"($w_q = {params.wq}$, perturbación inicial del 5%)",
                 fontsize=11, y=0.995)

    fig.savefig(save_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en {save_path}.{{png,pdf}}")


def hopf_eigenvalue_sweep(wq_range=(1e-6, 1e-1), n=300):
    """Barrido de autovalores del jacobiano vs wq en escala logarítmica.

    Devuelve:
        wqs: np.ndarray, valores de wq.
        re_max: np.ndarray, parte real máxima de los autovalores.
        omega: np.ndarray, frecuencia angular del par complejo dominante.
        wq_c: float, valor crítico donde Re_max cambia de signo (None si no hay cruce).
    """
    wqs = np.logspace(np.log10(wq_range[0]), np.log10(wq_range[1]), n)
    re_max = np.zeros(n)
    omega = np.full(n, np.nan)

    for i, wq in enumerate(wqs):
        stab = linear_stability(MGTParams(wq=wq))
        re_max[i] = stab["re_max"]
        if stab["omega_dominant"] is not None:
            omega[i] = stab["omega_dominant"]

    # Localizar wq_c por bisección si hay cruce.
    wq_c = None
    sign_change = np.where(np.diff(np.sign(re_max)) != 0)[0]
    if len(sign_change) > 0:
        idx = sign_change[0]
        lo, hi = wqs[idx], wqs[idx + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if linear_stability(MGTParams(wq=mid))["re_max"] > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        wq_c = 0.5 * (lo + hi)

    return wqs, re_max, omega, wq_c


def bifurcation_diagram(wq_values, t_final=300.0, h=1e-3, settling_fraction=0.6):
    """Para cada wq, integra hasta t_final con RK4 y mide amplitud pico-a-pico
    de q en la última fracción (1 - settling_fraction) del horizonte.

    Devuelve:
        wq_values: np.ndarray
        amp_q: np.ndarray, amplitud pico-a-pico de q en el atractor.
        q_mean: np.ndarray, valor medio de q en el atractor.
    """
    from integrators import rk4

    amp_q = np.zeros(len(wq_values))
    q_mean = np.zeros(len(wq_values))

    for i, wq in enumerate(wq_values):
        params = MGTParams(wq=wq)
        eq = equilibrium(params)
        # CI ligeramente perturbada para excitar el modo inestable.
        y0 = np.array([eq["W"] * 0.95, eq["q"] * 0.95, eq["x"] * 0.95])
        t, y = rk4(f, (0.0, t_final), y0, h, args=(params,))

        idx_start = int(len(t) * settling_fraction)
        q_ss = y[idx_start:, 1]
        amp_q[i] = q_ss.max() - q_ss.min()
        q_mean[i] = q_ss.mean()

    return wq_values, amp_q, q_mean


def make_figure_3(save_path):
    """Figura 3: bifurcación de Hopf supercrítica al variar wq.

    (a) q(t) en régimen oscilatorio (post-Hopf): ciclo límite (zoom).
    (b) Plano de fases (W, q) mostrando el ciclo cerrado.
    (c) Re(lambda_max) vs wq del análisis lineal.
    (d) Diagrama de bifurcación: amplitud pico-a-pico de q vs. wq, con
        ajuste de la raíz cuadrada característica de una Hopf supercrítica
        cerca del punto crítico.
    """
    from integrators import rk4

    # Análisis lineal: localizar wq_c.
    wqs_lin, re_max, omega_lin, wq_c = hopf_eigenvalue_sweep()
    print(f"\nFigura 3 - Análisis de Hopf:")
    print(f"  wq_c (bisección) = {wq_c:.4e}")
    stab_c = linear_stability(MGTParams(wq=wq_c))
    T_c = 2 * np.pi / stab_c["omega_dominant"]
    print(f"  omega en el cruce = {stab_c['omega_dominant']:.4f} rad/s, "
          f"T = {T_c:.4f} s")

    # Simulación post-Hopf para los paneles (a) y (b).
    # Usamos wq apenas por debajo de wq_c para que el ciclo sea de amplitud
    # moderada y el período observado se compare bien con la predicción lineal.
    wq_post = 8e-3  # ~75% de wq_c.
    params_post = MGTParams(wq=wq_post)
    eq_post = equilibrium(params_post)
    y0_post = np.array([eq_post["W"] * 0.95, eq_post["q"] * 0.95,
                        eq_post["x"] * 0.95])
    t_post, y_post = rk4(f, (0.0, 80.0), y0_post, 1e-3, args=(params_post,))
    W_post, q_post, x_post = y_post[:, 0], y_post[:, 1], y_post[:, 2]

    # Período observado: zero-crossings ascendentes de q-q_mean en estado est.
    idx_ss = int(len(t_post) * 0.6)
    q_centered = q_post[idx_ss:] - q_post[idx_ss:].mean()
    zero_cross = np.where(np.diff(np.sign(q_centered)) > 0)[0]
    T_obs = None
    if len(zero_cross) >= 2:
        T_obs = (t_post[idx_ss + zero_cross[-1]]
                 - t_post[idx_ss + zero_cross[0]]) / (len(zero_cross) - 1)
        T_pred = 2 * np.pi / linear_stability(params_post)["omega_dominant"]
        print(f"  Período observado (wq={wq_post}) = {T_obs:.4f} s")
        print(f"  Período del análisis lineal en wq={wq_post}: {T_pred:.4f} s")
        print(f"  Período en wq_c (cruce): {T_c:.4f} s")

    # Barrido para el diagrama de bifurcación.
    print("  Generando diagrama de bifurcación (barrido en wq)...")
    wq_sweep = np.concatenate([
        np.linspace(2e-3, wq_c - 5e-5, 30),     # Post-Hopf, denso.
        np.linspace(wq_c + 5e-5, 2.5e-2, 12),   # Pre-Hopf.
    ])
    wq_sweep = np.sort(wq_sweep)
    wq_bif, amp_q_bif, q_mean_bif = bifurcation_diagram(
        wq_sweep, t_final=400.0, h=2e-3, settling_fraction=0.7,
    )

    # Ajuste de raíz cuadrada A ~ K*sqrt(wq_c - wq) en la rama post-Hopf
    # CERCA del punto crítico (sólo puntos no saturados).
    mask_fit = (wq_bif < wq_c) & (amp_q_bif < 200) & (amp_q_bif > 5)
    if mask_fit.sum() >= 3:
        dist = wq_c - wq_bif[mask_fit]
        # A = K * sqrt(dist), ajuste por mínimos cuadrados sobre A^2 vs dist.
        K_squared = np.sum(amp_q_bif[mask_fit]**2 * dist) / np.sum(dist**2)
        K_fit = np.sqrt(max(K_squared, 0))
        print(f"  Ajuste supercrítico: A ~ {K_fit:.2f} * sqrt(wq_c - wq)")
    else:
        K_fit = None

    # ---- Layout ----
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32,
                          height_ratios=[1, 1.1])

    # (a) q(t) post-Hopf - zoom temporal a los últimos ~10 períodos.
    ax_q = fig.add_subplot(gs[0, 0])
    # Zoom: últimos 12 s del horizonte (régimen estacionario).
    mask_zoom = t_post > t_post[-1] - 12.0
    ax_q.plot(t_post[mask_zoom], q_post[mask_zoom], color="#d62728",
              linewidth=1.3)
    ax_q.axhline(eq_post["q"], color="gray", linestyle="--", linewidth=0.9,
                 label=fr"$q^* = {eq_post['q']:.1f}$")
    if T_obs is not None:
        # Marcar un período observado visualmente.
        t0_mark = t_post[mask_zoom][2]
        ax_q.annotate("", xy=(t0_mark + T_obs, q_post[mask_zoom][2]),
                      xytext=(t0_mark, q_post[mask_zoom][2]),
                      arrowprops=dict(arrowstyle="<->", color="black", lw=1))
        ax_q.text(t0_mark + T_obs/2, q_post[mask_zoom].max() * 0.95,
                  fr"$T_{{obs}}={T_obs:.2f}$ s",
                  ha="center", fontsize=9)
    ax_q.set_xlabel(r"Tiempo $t$ [s]")
    ax_q.set_ylabel(r"Cola $q$ [pkt]")
    ax_q.set_title(fr"(a) Ciclo límite: $q(t)$, $w_q={wq_post}$ (zoom)")
    ax_q.legend(fontsize=9, loc="lower right")
    ax_q.grid(True, alpha=0.3)

    # (b) Plano de fases post-Hopf (solo estado estacionario).
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_phase.plot(W_post[idx_ss:], q_post[idx_ss:], color="#1f77b4",
                  linewidth=1.2, label="ciclo límite")
    ax_phase.plot(W_post[:idx_ss], q_post[:idx_ss], color="lightgray",
                  linewidth=0.6, alpha=0.5, label="transitorio")
    ax_phase.plot(eq_post["W"], eq_post["q"], "*", color="red", markersize=14,
                  label="Equilibrio (inestable)", zorder=5)
    ax_phase.set_xlabel(r"Ventana TCP $W$ [pkt]")
    ax_phase.set_ylabel(r"Cola $q$ [pkt]")
    ax_phase.set_title("(b) Plano de fases - ciclo límite")
    ax_phase.legend(fontsize=9, loc="upper right")
    ax_phase.grid(True, alpha=0.3)

    # (c) Re(lambda_max) vs wq.
    ax_eig = fig.add_subplot(gs[0, 2])
    ax_eig.semilogx(wqs_lin, re_max, color="black", linewidth=1.3)
    ax_eig.axhline(0, color="gray", linestyle="-", linewidth=0.7)
    ax_eig.axvline(wq_c, color="red", linestyle="--", linewidth=1.0,
                   label=fr"$w_q^c = {wq_c:.2e}$")
    ax_eig.fill_between(wqs_lin, 0, re_max, where=(re_max > 0),
                        color="red", alpha=0.15, label="inestable")
    ax_eig.fill_between(wqs_lin, 0, re_max, where=(re_max < 0),
                        color="green", alpha=0.15, label="estable")
    ax_eig.set_xlabel(r"Peso EWMA $w_q$")
    ax_eig.set_ylabel(r"$\mathrm{Re}(\lambda_{\max})$")
    ax_eig.set_title("(c) Análisis lineal")
    ax_eig.legend(fontsize=8, loc="upper right")
    ax_eig.grid(True, alpha=0.3, which="both")

    # (d) Diagrama de bifurcación (fila completa abajo).
    ax_bif = fig.add_subplot(gs[1, :])
    mask_post = wq_bif < wq_c
    ax_bif.semilogx(wq_bif[mask_post], amp_q_bif[mask_post], "o",
                    color="#d62728", markersize=7,
                    label=r"Post-Hopf: ciclo límite (simulación)")
    ax_bif.semilogx(wq_bif[~mask_post], amp_q_bif[~mask_post], "s",
                    color="#1f77b4", markersize=7,
                    label=r"Pre-Hopf: equilibrio estable (simulación)")

    # Ajuste teórico de raíz cuadrada.
    if K_fit is not None:
        wq_fit = np.linspace(wq_c - 5e-3, wq_c, 100)
        wq_fit = wq_fit[wq_fit < wq_c]
        amp_fit = K_fit * np.sqrt(wq_c - wq_fit)
        ax_bif.semilogx(wq_fit, amp_fit, "--", color="black", linewidth=1.4,
                        label=fr"Ajuste $A \sim {K_fit:.1f}"
                              fr"\sqrt{{w_q^c - w_q}}$ (supercrítica)")

    ax_bif.axvline(wq_c, color="black", linestyle=":", linewidth=1.0)
    ax_bif.set_xlabel(r"Peso EWMA $w_q$")
    ax_bif.set_ylabel(r"Amplitud pico-a-pico de $q$ [pkt]")
    ax_bif.set_title("(d) Diagrama de bifurcación supercrítica de Hopf")
    ax_bif.legend(fontsize=10, loc="upper right")
    ax_bif.grid(True, alpha=0.3, which="both")
    ax_bif.set_ylim(-15, max(amp_q_bif) * 1.05)

    fig.suptitle("Fig. 3 - Bifurcación de Hopf al variar el peso EWMA "
                 "$w_q$ del filtro RED", fontsize=11, y=0.995)

    fig.savefig(save_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en {save_path}.{{png,pdf}}")

    return wq_c, T_obs


def make_figure_4(save_path):
    """Figura 4: extensión a dos rutas con balanceo dinámico.

    (a) q1(t) y q2(t) en el mismo eje: ecualización de cargas.
    (b) phi(t) convergiendo al splitting de Wardrop.
    """
    from routing import TwoRouteParams, f_two_routes, equilibrium_two_routes
    from integrators import rk4

    params = TwoRouteParams()
    eq = equilibrium_two_routes(params)
    print(f"\nFigura 4 - Dos rutas con balanceo dinámico:")
    print(f"  C1={params.route1.C}, C2={params.route2.C}, "
          f"N={params.N}, alpha={params.alpha}")
    print(f"  phi* (Wardrop) = {eq['phi']:.6f}")
    print(f"  q1* = q2* = {eq['q1']:.2f} pkt")
    print(f"  p1* = p2* = {eq['p1']:.4e}")

    # CI: phi=0.5 (balanceo neutro), rutas en sus equilibrios respectivos.
    y0 = np.array([
        eq["W1"] * 0.9, eq["q1"] * 0.9, eq["x1"] * 0.9,
        eq["W2"] * 0.9, eq["q2"] * 0.9, eq["x2"] * 0.9,
        0.5,
    ])
    t_span = (0.0, 60.0)
    t, y = rk4(f_two_routes, t_span, y0, 1e-3, args=(params,))

    W1, q1, x1 = y[:, 0], y[:, 1], y[:, 2]
    W2, q2, x2 = y[:, 3], y[:, 4], y[:, 5]
    phi = y[:, 6]

    err_phi = abs(phi[-1] - eq["phi"]) / eq["phi"] * 100
    print(f"  phi final = {phi[-1]:.6f}  (error rel = {err_phi:.4f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # (a) q1(t) y q2(t).
    ax = axes[0]
    ax.plot(t, q1, color="#1f77b4", linewidth=1.2,
            label=fr"$q_1(t)$, $C_1 = {params.route1.C:.0f}$ pkt/s")
    ax.plot(t, q2, color="#d62728", linewidth=1.2,
            label=fr"$q_2(t)$, $C_2 = {params.route2.C:.0f}$ pkt/s")
    ax.axhline(eq["q1"], color="gray", linestyle="--", linewidth=0.8,
               label=fr"$q_1^* = q_2^* = {eq['q1']:.1f}$ (Wardrop)")
    ax.set_xlabel(r"Tiempo $t$ [s]")
    ax.set_ylabel(r"Cola [pkt]")
    ax.set_title("(a) Ecualización de cargas entre rutas")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # (b) phi(t).
    ax = axes[1]
    ax.plot(t, phi, color="#2ca02c", linewidth=1.5)
    ax.axhline(eq["phi"], color="gray", linestyle="--", linewidth=0.9,
               label=fr"$\varphi^* = {eq['phi']:.3f} = "
                     fr"C_1/(C_1+C_2)$")
    ax.axhline(0.5, color="lightgray", linestyle=":", linewidth=0.9,
               label=r"$\varphi_0 = 0.5$ (CI: balanceo neutro)")
    ax.set_xlabel(r"Tiempo $t$ [s]")
    ax.set_ylabel(r"Fracción de tráfico por ruta 1, $\varphi$")
    ax.set_title("(b) Convergencia al splitting de Wardrop")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 0.65)

    fig.suptitle("Fig. 4 - Extensión a dos rutas paralelas: "
                 fr"balanceo dinámico ($\alpha = {params.alpha}$)",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    fig.savefig(save_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en {save_path}.{{png,pdf}}")


if __name__ == "__main__":
    print("Ejecutando estudio de convergencia principal...")
    res_main, params, eq, y0, t_span = run_main_study()

    print_text_table(res_main)
    print_latex_table(res_main)

    fig_path = os.path.join(FIGS_DIR, "fig1_convergence")
    make_figure_1(res_main, fig_path)

    print("\nEjecutando demostración del límite de estabilidad...")
    res_demo, hs_demo = run_stability_demo()
    print_stability_demo(res_demo, hs_demo)

    print("\nGenerando Figura 2 (régimen estable)...")
    fig2_path = os.path.join(FIGS_DIR, "fig2_stable")
    make_figure_2(fig2_path)

    print("\nGenerando Figura 3 (bifurcación de Hopf)...")
    fig3_path = os.path.join(FIGS_DIR, "fig3_hopf")
    make_figure_3(fig3_path)

    print("\nGenerando Figura 4 (dos rutas con balanceo)...")
    fig4_path = os.path.join(FIGS_DIR, "fig4_routing")
    make_figure_4(fig4_path)
