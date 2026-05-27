"""
Animaciones del modelo MGT y su extensión a dos rutas.

Genera dos archivos MP4 en anim/:
    - anim1_hopf.mp4: transición régimen estable -> ciclo límite al
      cambiar el peso EWMA wq.
    - anim2_routing.mp4: balanceo dinámico entre dos rutas paralelas con
      capacidades distintas, mostrando convergencia al splitting de Wardrop.

Requiere ffmpeg instalado en el sistema (matplotlib lo invoca).

Uso:
    python -m src.animations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection

from mgt_model import MGTParams, f, equilibrium
from routing import TwoRouteParams, f_two_routes, equilibrium_two_routes
from integrators import rk4


ANIM_DIR = os.path.join(os.path.dirname(__file__), "..", "anim")
os.makedirs(ANIM_DIR, exist_ok=True)

# Parámetros globales de render.
FPS = 30
DPI = 120


def _select_writer(save_path):
    """Selecciona un writer compatible con el entorno actual.

    Si ffmpeg está disponible, conserva MP4. Si no, usa PillowWriter y
    cambia la extensión a GIF para evitar fallos por dependencia externa.
    """
    if FFMpegWriter.isAvailable():
        return FFMpegWriter(fps=FPS, bitrate=2400), save_path

    base, _ = os.path.splitext(save_path)
    fallback_path = base + ".gif"
    print("  Aviso: ffmpeg no está disponible; se usará PillowWriter y GIF.")
    print(f"  La salida se guardará en {fallback_path}")
    return PillowWriter(fps=FPS), fallback_path


def animate_hopf(save_path, duration_s=20.0):
    """Animación 1: transición régimen estable -> ciclo límite.

    Estructura:
        - Primera mitad: simulación con wq=5e-2 (régimen estable).
        - Etiqueta intermedia.
        - Segunda mitad: simulación con wq=5e-3 (post-Hopf).

    Tres paneles: W(t), q(t), plano de fases (W, q) animados en sincronía.
    """
    # ---- Simulaciones de los dos regímenes ----
    # 1) Régimen estable: wq=5e-2 (W* y q* en el equilibrio).
    params_stable = MGTParams(wq=5e-2)
    eq_st = equilibrium(params_stable)
    y0_st = np.array([eq_st["W"] * 0.95, eq_st["q"] * 0.95, eq_st["x"] * 0.95])
    t_st, y_st = rk4(f, (0.0, 30.0), y0_st, 1e-3, args=(params_stable,))

    # 2) Régimen oscilatorio: wq=5e-3 (post-Hopf).
    params_osc = MGTParams(wq=5e-3)
    eq_osc = equilibrium(params_osc)
    y0_osc = np.array([eq_osc["W"] * 0.95, eq_osc["q"] * 0.95, eq_osc["x"] * 0.95])
    t_osc, y_osc = rk4(f, (0.0, 30.0), y0_osc, 1e-3, args=(params_osc,))

    # Submuestreo para animación fluida: queremos n_frames totales.
    n_frames = int(FPS * duration_s)
    frames_per_phase = n_frames // 2

    # Submuestrear cada fase para tener frames_per_phase puntos.
    def subsample(t, y, k):
        idx = np.linspace(0, len(t) - 1, k, dtype=int)
        return t[idx], y[idx]

    t_st_s, y_st_s = subsample(t_st, y_st, frames_per_phase)
    t_osc_s, y_osc_s = subsample(t_osc, y_osc, frames_per_phase)

    # ---- Figura con tres paneles ----
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.2],
                          hspace=0.4, wspace=0.28)

    ax_W = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[1, 0])
    ax_phase = fig.add_subplot(gs[:, 1])

    # Límites de los ejes (fijos durante toda la animación).
    W_min = min(y_st[:, 0].min(), y_osc[:, 0].min()) - 0.5
    W_max = max(y_st[:, 0].max(), y_osc[:, 0].max()) + 0.5
    q_min = min(y_st[:, 1].min(), y_osc[:, 1].min()) - 10
    q_max = max(y_st[:, 1].max(), y_osc[:, 1].max()) + 10

    for ax, ylabel, ymin, ymax in [
        (ax_W, r"Ventana TCP $W$ [pkt]", W_min, W_max),
        (ax_q, r"Cola $q$ [pkt]", q_min, q_max),
    ]:
        ax.set_xlim(0, 30)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"Tiempo $t$ [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    ax_phase.set_xlim(W_min, W_max)
    ax_phase.set_ylim(q_min, q_max)
    ax_phase.set_xlabel(r"$W$ [pkt]")
    ax_phase.set_ylabel(r"$q$ [pkt]")
    ax_phase.set_title("Plano de fases")
    ax_phase.grid(True, alpha=0.3)

    # Líneas (vacías al principio, se llenan en cada frame).
    line_W, = ax_W.plot([], [], color="#1f77b4", linewidth=1.4)
    line_q, = ax_q.plot([], [], color="#d62728", linewidth=1.4)
    line_phase, = ax_phase.plot([], [], color="#2ca02c", linewidth=1.2)
    point_phase, = ax_phase.plot([], [], "o", color="black", markersize=6)

    # Marcas de los equilibrios (cambian entre fases).
    eq_W_line = ax_W.axhline(eq_st["W"], color="gray", linestyle="--",
                              linewidth=0.9)
    eq_q_line = ax_q.axhline(eq_st["q"], color="gray", linestyle="--",
                              linewidth=0.9)
    eq_phase_point, = ax_phase.plot(eq_st["W"], eq_st["q"], "*",
                                    color="red", markersize=14, zorder=5)

    # Título dinámico.
    title = fig.suptitle("", fontsize=12, y=0.98)

    def init():
        return (line_W, line_q, line_phase, point_phase,
                eq_W_line, eq_q_line, eq_phase_point, title)

    def update(frame):
        if frame < frames_per_phase:
            # Fase 1: régimen estable.
            t_arr, y_arr = t_st_s, y_st_s
            i = frame
            eq_use = eq_st
            phase_label = (fr"Fase 1: $w_q = {params_stable.wq}$ - "
                           r"$\mathrm{Re}(\lambda_{\max}) < 0$ "
                           r"$\Rightarrow$ equilibrio estable")
            line_phase.set_color("#1f77b4")
        else:
            # Fase 2: régimen post-Hopf.
            t_arr, y_arr = t_osc_s, y_osc_s
            i = frame - frames_per_phase
            eq_use = eq_osc
            phase_label = (fr"Fase 2: $w_q = {params_osc.wq}$ "
                           fr"(< $w_q^c$) - "
                           r"$\mathrm{Re}(\lambda_{\max}) > 0$ "
                           r"$\Rightarrow$ ciclo límite estable")
            line_phase.set_color("#d62728")

        # Actualizar series temporales.
        line_W.set_data(t_arr[: i + 1], y_arr[: i + 1, 0])
        line_q.set_data(t_arr[: i + 1], y_arr[: i + 1, 1])

        # Trayectoria en plano de fases.
        line_phase.set_data(y_arr[: i + 1, 0], y_arr[: i + 1, 1])
        if i >= 0:
            point_phase.set_data([y_arr[i, 0]], [y_arr[i, 1]])

        # Actualizar líneas de equilibrio (saltan al cambiar de fase).
        eq_W_line.set_ydata([eq_use["W"], eq_use["W"]])
        eq_q_line.set_ydata([eq_use["q"], eq_use["q"]])
        eq_phase_point.set_data([eq_use["W"]], [eq_use["q"]])

        title.set_text(phase_label)

        return (line_W, line_q, line_phase, point_phase,
                eq_W_line, eq_q_line, eq_phase_point, title)

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=1000 / FPS, blit=False)

    print(f"  Renderizando {n_frames} frames a {FPS} fps...")
    writer, output_path = _select_writer(save_path)
    anim.save(output_path, writer=writer, dpi=DPI)
    plt.close(fig)
    print(f"  Animación guardada en {output_path}")


def animate_routing(save_path, duration_s=15.0):
    """Animación 2: balanceo dinámico de dos rutas.

    Layout:
        Arriba-izq:  q1(t) y q2(t) (ecualización de cargas).
        Arriba-der:  phi(t) (convergencia a Wardrop).
        Abajo:       dos colas verticales como "barras" con altura q_k(t)
                     y ancho proporcional a phi (ruta 1) o 1-phi (ruta 2),
                     visualización tipo "bufferbloat".
    """
    params = TwoRouteParams()
    eq = equilibrium_two_routes(params)

    # CI con balanceo neutro.
    y0 = np.array([
        eq["W1"] * 0.9, eq["q1"] * 0.9, eq["x1"] * 0.9,
        eq["W2"] * 0.9, eq["q2"] * 0.9, eq["x2"] * 0.9,
        0.5,
    ])
    t_full, y_full = rk4(f_two_routes, (0.0, 60.0), y0, 1e-3, args=(params,))

    # Submuestreo a n_frames.
    n_frames = int(FPS * duration_s)
    idx = np.linspace(0, len(t_full) - 1, n_frames, dtype=int)
    t = t_full[idx]
    y = y_full[idx]

    # ---- Figura ----
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.4, wspace=0.28)

    ax_q = fig.add_subplot(gs[0, 0])
    ax_phi = fig.add_subplot(gs[0, 1])
    ax_bars = fig.add_subplot(gs[1, :])

    # Series temporales (q1, q2).
    ax_q.set_xlim(0, t[-1])
    ax_q.set_ylim(0, max(y[:, 1].max(), y[:, 4].max()) * 1.05)
    ax_q.set_xlabel(r"$t$ [s]")
    ax_q.set_ylabel(r"Cola [pkt]")
    ax_q.set_title("Ecualización de cargas $q_1, q_2$")
    ax_q.axhline(eq["q1"], color="gray", linestyle="--", linewidth=0.9,
                 label=fr"$q^* = {eq['q1']:.1f}$ (Wardrop)")
    ax_q.grid(True, alpha=0.3)
    line_q1, = ax_q.plot([], [], color="#1f77b4", linewidth=1.4, label=r"$q_1$")
    line_q2, = ax_q.plot([], [], color="#d62728", linewidth=1.4, label=r"$q_2$")
    ax_q.legend(fontsize=9, loc="lower right")

    # phi(t).
    ax_phi.set_xlim(0, t[-1])
    ax_phi.set_ylim(0.4, 0.7)
    ax_phi.set_xlabel(r"$t$ [s]")
    ax_phi.set_ylabel(r"$\varphi$")
    ax_phi.set_title("Fracción de tráfico por ruta 1")
    ax_phi.axhline(eq["phi"], color="gray", linestyle="--", linewidth=0.9,
                   label=fr"$\varphi^* = {eq['phi']:.3f}$")
    ax_phi.axhline(0.5, color="lightgray", linestyle=":", linewidth=0.9)
    ax_phi.grid(True, alpha=0.3)
    line_phi, = ax_phi.plot([], [], color="#2ca02c", linewidth=1.6)
    ax_phi.legend(fontsize=9, loc="lower right")

    # Panel de barras: visualización física tipo "buffer fillómetro".
    ax_bars.set_xlim(-0.5, 2.5)
    ax_bars.set_ylim(0, max(y[:, 1].max(), y[:, 4].max()) * 1.1)
    ax_bars.set_xticks([0.5, 1.7])
    ax_bars.set_xticklabels([fr"Ruta 1: $C_1 = {params.route1.C:.0f}$",
                              fr"Ruta 2: $C_2 = {params.route2.C:.0f}$"])
    ax_bars.set_ylabel(r"Llenado de cola [pkt]")
    ax_bars.set_title("Visualización física: las colas se llenan y vacían "
                      "mientras $\\varphi$ ajusta el reparto")
    ax_bars.grid(True, alpha=0.3, axis="y")

    # Barras: ancho proporcional a phi y 1-phi.
    bar1 = ax_bars.bar([0.5], [0], width=0.4, color="#1f77b4",
                       edgecolor="black", linewidth=1.2, label="Ruta 1")
    bar2 = ax_bars.bar([1.7], [0], width=0.4, color="#d62728",
                       edgecolor="black", linewidth=1.2, label="Ruta 2")

    # Texto con valores numéricos en cada barra.
    txt1 = ax_bars.text(0.5, 5, "", ha="center", fontsize=10, fontweight="bold")
    txt2 = ax_bars.text(1.7, 5, "", ha="center", fontsize=10, fontweight="bold")
    txt_time = ax_bars.text(0.05, 0.95, "", transform=ax_bars.transAxes,
                            fontsize=11, verticalalignment="top",
                            bbox=dict(facecolor="white", alpha=0.8,
                                      edgecolor="gray"))

    def init():
        return (line_q1, line_q2, line_phi, bar1, bar2, txt1, txt2, txt_time)

    def update(frame):
        line_q1.set_data(t[: frame + 1], y[: frame + 1, 1])
        line_q2.set_data(t[: frame + 1], y[: frame + 1, 4])
        line_phi.set_data(t[: frame + 1], y[: frame + 1, 6])

        q1_now = max(0.0, y[frame, 1])
        q2_now = max(0.0, y[frame, 4])
        phi_now = y[frame, 6]

        # Barras: la altura es la cola; el ancho es proporcional a phi.
        bar1[0].set_height(q1_now)
        bar1[0].set_width(0.15 + 0.35 * phi_now)
        bar1[0].set_x(0.5 - bar1[0].get_width() / 2)

        bar2[0].set_height(q2_now)
        bar2[0].set_width(0.15 + 0.35 * (1.0 - phi_now))
        bar2[0].set_x(1.7 - bar2[0].get_width() / 2)

        txt1.set_text(f"q1 = {q1_now:.0f}\nN1 = {params.N * phi_now:.1f}")
        txt2.set_text(f"q2 = {q2_now:.0f}\nN2 = {params.N * (1-phi_now):.1f}")
        txt_time.set_text(fr"$t = {t[frame]:.1f}$ s    "
                          fr"$\varphi = {phi_now:.3f}$")

        return (line_q1, line_q2, line_phi, bar1, bar2, txt1, txt2, txt_time)

    fig.suptitle("Balanceo dinámico de dos rutas paralelas - "
                 "convergencia al splitting de Wardrop", fontsize=12, y=0.99)

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=1000 / FPS, blit=False)

    print(f"  Renderizando {n_frames} frames a {FPS} fps...")
    writer, output_path = _select_writer(save_path)
    anim.save(output_path, writer=writer, dpi=DPI)
    plt.close(fig)
    print(f"  Animación guardada en {output_path}")


if __name__ == "__main__":
    print("Generando Animación 1 (bifurcación de Hopf)...")
    animate_hopf(os.path.join(ANIM_DIR, "anim1_hopf.mp4"), duration_s=20.0)

    print("\nGenerando Animación 2 (dos rutas con balanceo)...")
    animate_routing(os.path.join(ANIM_DIR, "anim2_routing.mp4"),
                    duration_s=15.0)

    print("\nListo. Sube los MP4 a YouTube no listado y actualiza el README.")
