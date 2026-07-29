"""Mide la distribución REAL de `radii` (sin el clamp a 50) sobre un modelo entrenado.

Carga el ply de un run, renderiza N cámaras de train y acumula max_radii2D igual que
train.py:214, pero con radius_clip desactivado -> dice cuánto mide de verdad lo que el
prune por pantalla ve saturado en 50.
"""
import sys, types, torch
sys.path.insert(0, "/home/jesus/Documents/Gaussian_splatting/2d-gaussian-splatting-beta-gabor")
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import GaussianModel

# --- copia del módulo del renderer con el clamp desactivado -------------------
src = open("gaussian_renderer/__init__.py").read()
assert "radius_clip = 50.0" in src
src_noclip = src.replace("radius_clip = 50.0", "radius_clip = 1e9")
mod = types.ModuleType("gaussian_renderer_noclip")
mod.__dict__["__file__"] = "gaussian_renderer/__init__.py"
exec(compile(src_noclip, "gaussian_renderer/__init__.py", "exec"), mod.__dict__)
render_noclip = mod.render

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=30000, type=int)
parser.add_argument("--n_cams", default=30, type=int)
args = get_combined_args(parser)
dataset, pipe = model.extract(args), pipeline.extract(args)

gaussians = GaussianModel(dataset.sh_degree, getattr(dataset, "sb_number", 0))
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

cams = scene.getTrainCameras()
step = max(1, len(cams) // args.n_cams)
cams = cams[::step][:args.n_cams]
N = gaussians.get_xyz.shape[0]
print(f"modelo: {args.model_path} iter={args.iteration}  N={N}  cams={len(cams)} "
      f"({int(cams[0].image_width)}x{int(cams[0].image_height)})")

acum = torch.zeros(N, device="cuda")
por_vista = []
with torch.no_grad():
    for i, cam in enumerate(cams):
        pkg = render_noclip(cam, gaussians, pipe, bg)
        radii, vis = pkg["radii"].float(), pkg["visibility_filter"]
        acum[vis] = torch.max(acum[vis], radii[vis])
        r = radii[vis]
        por_vista.append((int(vis.sum()), float(r.max()), float((r > 20).float().mean() * 100),
                          float((r >= 50).float().mean() * 100)))
        del pkg
        torch.cuda.empty_cache()

print("\n--- por vista (splats visibles, radio max, %>20px, %>=50px) ---")
for i, (nv, mx, p20, p50) in enumerate(por_vista[:8]):
    print(f"  vista {i:2d}: visibles={nv:>9,}  max={mx:>7.0f}px  >20px={p20:5.2f}%  >=50px={p50:5.2f}%")

vistos = acum > 0
a = acum[vistos]
qs = torch.quantile(a[torch.randperm(a.numel(), device=a.device)[:1_000_000]],
                    torch.tensor([0.5, 0.9, 0.99, 0.999], device=a.device))
print(f"\n--- max_radii2D acumulado sobre {len(cams)} vistas ---")
print(f"  splats vistos al menos una vez: {int(vistos.sum()):,} de {N:,}")
print(f"  mediana={qs[0]:.1f}px  p90={qs[1]:.1f}px  p99={qs[2]:.1f}px  p99.9={qs[3]:.1f}px  "
      f"MAX={a.max():.0f}px")
for thr in (20, 50, 100, 200, 400, 800):
    print(f"  radio > {thr:>4} px: {int((a > thr).sum()):>9,}  ({100*float((a>thr).float().mean()):6.2f}% de los vistos)")
sat = (a >= 50).sum()
print(f"\n  el clamp a 50 aplasta {int(sat):,} splats ({100*float((a>=50).float().mean()):.2f}% de los vistos) "
      f"en un solo valor; su radio real va de 50 a {a.max():.0f} px")
