# -*- coding: utf-8 -*-
"""Juguete que reproduce EXACTAMENTE la lógica de los dos prunes de tamaño
   de scene/gaussian_model.py, con 6 splats y a mano, para ver dónde se rompe."""
import torch
torch.manual_seed(0)          # reproducible: las cifras coinciden con docs/prunes_de_tamano_explicado.html

EXTENT = 10.0          # spatial_lr_scale de la escena (aquí, redondo)
CLAMP  = 0.1           # scale_clamp_factor
TECHO  = CLAMP * EXTENT        # 1.0
MAX_SCREEN = 20        # size_threshold que pasa train.py

NOMBRES = ["A normal", "B normal", "C GIGANTE mundo", "D GIGANTE pantalla", "E normal", "F GIGANTE ambos"]
#  escala CRUDA en el mundo (antes del clamp)      radio en pantalla observado (px)
S_CRUDA = torch.tensor([0.20, 0.35, 4.00, 0.30, 0.15, 3.00])
R_PANT  = torch.tensor([ 4.0,  7.0,  12.0, 45.0,  3.0, 60.0])

def get_scaling(s_cruda):
    """Réplica de gaussian_model.py:132-141 — la activación YA viene clampada."""
    return s_cruda.clamp(max=TECHO)

def seccion(t): print("\n" + "="*74 + "\n" + t + "\n" + "="*74)

seccion("PASO 0 · Qué tenemos")
print(f"{'splat':<20}{'escala CRUDA':>14}{'get_scaling':>13}{'radio px':>10}")
for i,n in enumerate(NOMBRES):
    print(f"{n:<20}{S_CRUDA[i]:>14.2f}{get_scaling(S_CRUDA)[i]:>13.2f}{R_PANT[i]:>10.1f}")
print(f"\ntecho de escala = scale_clamp_factor x extent = {CLAMP} x {EXTENT} = {TECHO}")
print(f"umbral de pantalla (size_threshold) = {MAX_SCREEN} px")
print("\nA OJO: hay que podar C (mundo), D (pantalla) y F (los dos). Deben sobrevivir A, B, E.")

# ─────────────────────────────────────────────────────────────────────────────
seccion("PASO 1 · Se acumula max_radii2D durante 100 iteraciones (train.py:214)")
max_radii2D = torch.zeros(6)
for it in range(1, 101):
    radii = R_PANT * (0.9 + 0.2*torch.rand(6))     # el radio fluctúa con la vista
    visible = torch.ones(6, dtype=torch.bool)
    max_radii2D[visible] = torch.max(max_radii2D[visible], radii[visible])
print("max_radii2D tras 100 iters:", [f"{v:.1f}" for v in max_radii2D])
print("-> el acumulador FUNCIONA: registra el radio máximo visto por cada splat.")

# ─────────────────────────────────────────────────────────────────────────────
seccion("PASO 2 · Entramos en densify_and_prune... y pasa esto")

def densification_postfix(max_radii2D, n_total):
    """Réplica de gaussian_model.py:629."""
    return torch.zeros(n_total)          # <-- LA LÍNEA CULPABLE

print("estado al ENTRAR en densify_and_prune:")
print("   max_radii2D =", [f"{v:.1f}" for v in max_radii2D])

print("\n  1) densify_and_clone(...)  -> llama a densification_postfix")
max_radii2D = densification_postfix(max_radii2D, 6)
print("     max_radii2D =", [f"{v:.1f}" for v in max_radii2D])

print("\n  2) densify_and_split(...)  -> llama a densification_postfix")
max_radii2D = densification_postfix(max_radii2D, 6)
print("     max_radii2D =", [f"{v:.1f}" for v in max_radii2D])

print("\n  3) AHORA se leen los dos prunes de tamaño (gaussian_model.py:816-819):")
big_points_vs = max_radii2D > MAX_SCREEN
big_points_ws = get_scaling(S_CRUDA).max() > 0.1 * EXTENT   # nota: .max() sobre las 2 escalas
big_points_ws = get_scaling(S_CRUDA) > 0.1 * EXTENT
print(f"     big_points_vs = max_radii2D > {MAX_SCREEN}      -> {big_points_vs.tolist()}")
print(f"     big_points_ws = get_scaling > {0.1*EXTENT}       -> {big_points_ws.tolist()}")
print(f"\n     PODADOS: {int((big_points_vs|big_points_ws).sum())} de 6.   ESPERADO: 3 (C, D, F)")

# ─────────────────────────────────────────────────────────────────────────────
seccion("PASO 3 · Por qué big_points_ws no dispara NUNCA (aritmética pura)")
gs = get_scaling(S_CRUDA)
print(f"{'splat':<20}{'cruda':>8}{'get_scaling':>13}{'> 1.0?':>9}")
for i,n in enumerate(NOMBRES):
    print(f"{n:<20}{S_CRUDA[i]:>8.2f}{gs[i]:>13.6f}{str(bool(gs[i] > TECHO)):>9}")
print(f"\nel MAYOR valor que get_scaling puede devolver es {gs.max():.6f} = el techo EXACTO.")
print(f"y se compara con '> {TECHO}'  (estricto).  ->  clamp(x, max=C) > C  es FALSE SIEMPRE.")

# ─────────────────────────────────────────────────────────────────────────────
seccion("PASO 4 · Con el FIX aplicado")
FACTOR = 0.7
def prune_arreglado(max_radii2D_guardado, s_cruda):
    vs = max_radii2D_guardado > MAX_SCREEN                       # FIX A: leer el valor GUARDADO
    ws = get_scaling(s_cruda) > FACTOR * CLAMP * EXTENT          # FIX B: umbral por DEBAJO del techo
    return vs, ws

max_radii2D = torch.zeros(6)                       # re-acumulamos, como en un run real
for it in range(100):
    max_radii2D = torch.max(max_radii2D, R_PANT * (0.9 + 0.2*torch.rand(6)))
guardado = max_radii2D.clone()                     # <-- FIX A: copia ANTES de clone/split
max_radii2D = densification_postfix(max_radii2D, 6)   # clone
max_radii2D = densification_postfix(max_radii2D, 6)   # split

vs, ws = prune_arreglado(guardado, S_CRUDA)
print(f"umbral de mundo con el fix: {FACTOR} x {TECHO} = {FACTOR*TECHO}")
print(f"\n{'splat':<20}{'radio px':>10}{'vs':>7}{'get_scaling':>13}{'ws':>7}{'PODADO':>9}")
for i,n in enumerate(NOMBRES):
    print(f"{n:<20}{guardado[i]:>10.1f}{str(bool(vs[i])):>7}{get_scaling(S_CRUDA)[i]:>13.2f}{str(bool(ws[i])):>7}{str(bool(vs[i] or ws[i])):>9}")
print(f"\nPODADOS: {int((vs|ws).sum())} de 6  ->  {[NOMBRES[i] for i in range(6) if (vs|ws)[i]]}")
print("ESPERADO: C, D, F.  ", "CORRECTO" if [NOMBRES[i] for i in range(6) if (vs|ws)[i]]==[NOMBRES[2],NOMBRES[3],NOMBRES[5]] else "MAL")
