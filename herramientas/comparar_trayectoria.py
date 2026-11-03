"""
herramientas/comparar_trayectoria.py — Trayectoria estimada frente a la verdad.

Uso, desde ~/svo-pi:
    python -m herramientas.comparar_trayectoria datos/sim_vio
Necesita en la carpeta: vio_trayectoria.csv (de apps/pi_vio) y verdad.csv (de una
grabación sintética). Guarda comparacion.png en la misma carpeta.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vio import so3
from vio.pipeline import leer_poses


def alinear(t_est, p_est, R_est, t_ver, p_ver, R_ver):
    """Expresa la verdad en el marco del filtro, haciendo coincidir la primera pose estimada.

    El filtro define su mundo al arrancar (origen donde está, rumbo 0), y la verdad tiene el
    suyo: se pasa de uno a otro con la pose verdadera en el instante de la primera estimación,
        R_alin = R_est[0]·R_ver(t0)ᵀ       p_ver_alin = R_alin·(p_ver - p_ver(t0)) + p_est[0]
    Devuelve: (p_ver_alin (N, 3), R_ver_alin (N, 3, 3)) en los instantes de la estimación.
    """
    indices = np.searchsorted(t_ver, t_est)               # la muestra de verdad de cada estimación
    indices = np.clip(indices, 0, len(t_ver) - 1)
    p_v, R_v = p_ver[indices], R_ver[indices]
    R_alin = R_est[0] @ R_v[0].T
    p_alin = (p_v - p_v[0]) @ R_alin.T + p_est[0]
    R_v_alin = np.array([R_alin @ R for R in R_v])
    return p_alin, R_v_alin


def main():
    parser = argparse.ArgumentParser(description="Compara vio_trayectoria.csv con verdad.csv")
    parser.add_argument("carpeta")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)

    t_est, p_est, R_est = leer_poses(carpeta / "vio_trayectoria.csv")
    t_ver, p_ver, R_ver = leer_poses(carpeta / "verdad.csv")
    p_v, R_v = alinear(t_est, p_est, R_est, t_ver, p_ver, R_ver)

    error_p = np.linalg.norm(p_est - p_v, axis=1)
    error_ang = np.degrees([np.linalg.norm(so3.log_so3(Rv.T @ Re)) for Re, Rv in zip(R_est, R_v)])
    distancia = np.linalg.norm(np.diff(p_v, axis=0), axis=1).sum()
    t = (t_est - t_est[0]) / 1e9
    print(f"{len(t_est)} poses, {t[-1]:.1f} s, {distancia:.1f} m recorridos")
    print(f"error de posición: final {error_p[-1]:.3f} m ({100 * error_p[-1] / distancia:.2f} % de la"
          f" distancia), medio {error_p.mean():.3f} m, máximo {error_p.max():.3f} m")
    print(f"error de orientación: final {error_ang[-1]:.2f}°, máximo {error_ang.max():.2f}°")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(p_v[:, 0], p_v[:, 1], "k", lw=2, label="verdad")
    ax1.plot(p_est[:, 0], p_est[:, 1], "b", lw=1, label="VIO")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [m] (marco del filtro)")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Vista desde arriba")
    ax1.legend()
    ax2.plot(t, error_p, label="posición [m]")
    ax2.plot(t, error_ang / 10, label="orientación [°/10]")
    ax2.set_xlabel("t desde el arranque del filtro [s]")
    ax2.set_title("Error")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "comparacion.png", dpi=90)
    print(f"Figura: {carpeta / 'comparacion.png'}")


if __name__ == "__main__":
    main()