"""Mide la forma REAL del kernel Gabor sobre el .ply de un run.

Para cada splat: a=(a1,a2,a3), beta.
  f(r)  = 1/2 + a1 cos(pi r) + a2 cos(3pi r) + a3 cos(5pi r)
  g(r)  = f(r)/f(0),  kernel(r) = g(r)^beta
Calcula:
  - histograma de sum(a)
  - contraste valle/pico del PRIMER minimo local (dianas)
  - % de splats NO monotonos (g'(r) > tol en algun punto de [0,1))
  - % "caja" (sum(a) ~ 0 -> kernel plano)
  - perfil medio del kernel
Salida: JSON a stdout.
"""
import sys, json
import numpy as np
from plyfile import PlyData

path = sys.argv[1]
sub = int(sys.argv[2]) if len(sys.argv) > 2 else 1

ply = PlyData.read(path)
v = ply["vertex"]
a1 = np.asarray(v["a_0"], dtype=np.float64)
a2 = np.asarray(v["a_1"], dtype=np.float64)
a3 = np.asarray(v["a_2"], dtype=np.float64)
beta_raw = np.asarray(v["beta"], dtype=np.float64)
opa_raw = np.asarray(v["opacity"], dtype=np.float64)
if sub > 1:
    a1, a2, a3, beta_raw, opa_raw = a1[::sub], a2[::sub], a3[::sub], beta_raw[::sub], opa_raw[::sub]

# get_a = clamp(min=0) ; get_beta = 4*exp(clamp(_beta,-4,2))
a1 = np.clip(a1, 0, None); a2 = np.clip(a2, 0, None); a3 = np.clip(a3, 0, None)
beta = 4.0 * np.exp(np.clip(beta_raw, -4.0, 2.0))
opa = 1.0 / (1.0 + np.exp(-opa_raw))
N = a1.size
S = a1 + a2 + a3

R = 257
r = np.linspace(0.0, 1.0, R)
C1 = np.cos(np.pi * r); C3 = np.cos(3 * np.pi * r); C5 = np.cos(5 * np.pi * r)

out = {"path": path, "N_total": int(len(np.asarray(v["a_0"]))), "N_muestra": int(N),
       "subsample": sub}
out["sum_a"] = {"mean": float(S.mean()), "p50": float(np.median(S)),
                "frac_frontera_0.5": float((S > 0.4999).mean()),
                "frac_menor_0.05_caja": float((S < 0.05).mean())}
out["a_mean"] = [float(a1.mean()), float(a2.mean()), float(a3.mean())]
out["beta"] = {"mean": float(beta.mean()), "p50": float(np.median(beta))}
out["opacity_mean"] = float(opa.mean())

# --- por bloques para no reventar memoria ---
BS = 200000
no_mono = 0
n_min2 = 0
contraste = []   # valle/pico del primer minimo local de g
perfil = np.zeros(R)
perfil_g = np.zeros(R)
hist_sum = np.zeros(50)
for i in range(0, N, BS):
    A1 = a1[i:i+BS][:, None]; A2 = a2[i:i+BS][:, None]; A3 = a3[i:i+BS][:, None]
    B = beta[i:i+BS][:, None]
    f = 0.5 + A1 * C1 + A2 * C3 + A3 * C5
    f0 = 0.5 + A1 + A2 + A3
    g = np.clip(f, 1e-6, None) / np.clip(f0, 1e-6, None)
    d = np.diff(g, axis=1)
    # no monotono: alguna subida apreciable
    no_mono += int((d.max(axis=1) > 1e-4).sum())
    # numero de minimos locales (cambio - -> +)
    signo = np.sign(d)
    cambios = ((signo[:, :-1] < 0) & (signo[:, 1:] > 0)).sum(axis=1)
    n_min2 += int((cambios >= 2).sum())
    # contraste: valor de g en el minimo global interior (excluyendo r=1)
    gm = g[:, :-8].min(axis=1)
    contraste.append(gm)
    k = np.clip(g, 1e-9, None) ** B
    perfil += k.sum(axis=0)
    perfil_g += g.sum(axis=0)
    hist_sum += np.histogram(S[i:i+BS], bins=50, range=(0, 0.5))[0]

contraste = np.concatenate(contraste)
out["no_monotonos_frac"] = no_mono / N
out["dos_o_mas_minimos_frac"] = n_min2 / N
out["contraste_valle_pico"] = {
    "mean": float(contraste.mean()), "p50": float(np.median(contraste)),
    "p10": float(np.percentile(contraste, 10)), "p90": float(np.percentile(contraste, 90)),
    "frac_menor_0.10": float((contraste < 0.10).mean()),
}
out["perfil_kernel_medio"] = [float(x) for x in (perfil / N)[::8]]
out["perfil_g_medio"] = [float(x) for x in (perfil_g / N)[::8]]
out["r_grid"] = [float(x) for x in r[::8]]
out["hist_sum_a"] = [int(x) for x in hist_sum]
print(json.dumps(out, indent=1))
