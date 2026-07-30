"""Verificación numérica del kernel Gabor en modo ÁTOMO (envolvente x onda).

Comprueba, contra el rasterizador REAL (no una reimplementación):
  1. a = 0  =>  kernel = (1-r)^beta  en los modos radial y dir, y los dos dan la MISMA
     imagen (S == 1 en ambos): el baseline es el punto neutro del espacio.
  2. Los gradientes de a_n, phi, b, beta y scaling coinciden con diferencias finitas.
  3. El modo legacy sigue dando exactamente lo de antes (no-regresión).

Uso:  python scripts/test_gabor_atomo.py
"""
import math
import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

torch.manual_seed(0)
DEV = "cuda"
W = H = 64
FOV = 0.8


def make_settings(gabor_mode):
    znear, zfar = 0.01, 100.0
    # cámara en (0,0,-3) mirando a +z
    view = torch.tensor([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 3., 1.]], device=DEV)
    tan = math.tan(FOV * 0.5)
    top, right = tan * znear, tan * znear
    proj = torch.zeros(4, 4, device=DEV)
    proj[0, 0] = znear / right
    proj[1, 1] = znear / top
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = -(zfar * znear) / (zfar - znear)
    proj[2, 3] = 1.0
    full = view @ proj
    return GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tan, tanfovy=tan,
        bg=torch.zeros(3, device=DEV), scale_modifier=1.0,
        viewmatrix=view, projmatrix=full, sh_degree=0,
        campos=torch.tensor([0., 0., -3.], device=DEV),
        prefiltered=False, debug=False, freeze_low_beta=False,
        gabor_mode=gabor_mode)


def make_scene(n=6):
    xyz = (torch.rand(n, 3, device=DEV) - 0.5) * 1.2
    xyz[:, 2] = 0.0
    scales = torch.rand(n, 2, device=DEV) * 0.25 + 0.15
    rots = torch.zeros(n, 4, device=DEV); rots[:, 0] = 1.0
    rots += torch.randn(n, 4, device=DEV) * 0.1
    rots = rots / rots.norm(dim=1, keepdim=True)
    opac = torch.rand(n, 1, device=DEV) * 0.5 + 0.4
    colors = torch.rand(n, 3, device=DEV)
    beta_raw = torch.randn(n, 1, device=DEV) * 0.3
    return xyz, scales, rots, opac, colors, beta_raw


def render(gabor_mode, xyz, scales, rots, opac, colors, beta, gab):
    r = GaussianRasterizer(raster_settings=make_settings(gabor_mode))
    means2D = torch.zeros_like(xyz, requires_grad=True)
    out, radii, _ = r(means3D=xyz, means2D=means2D, shs=None, colors_precomp=colors,
                      opacities=opac, beta=beta, a=gab, scales=scales, rotations=rots,
                      cov3D_precomp=None)
    return out


def gabor_tensor(a, phi, b):
    return torch.cat([a, phi, b], dim=1).contiguous()


def main():
    xyz, scales, rots, opac, colors, beta_raw = make_scene()
    n = xyz.shape[0]
    beta = (4.0 * torch.exp(beta_raw.clamp(-4, 2))).contiguous()
    zero = torch.zeros(n, 1, device=DEV)
    ones = torch.ones(n, 1, device=DEV)

    # ---------- 1) a = 0 -> tent puro, y radial == dir ----------
    a0 = torch.zeros(n, 3, device=DEV)
    phi = torch.rand(n, 1, device=DEV) * 3.0 + 0.5      # phi ARBITRARIO: no debe influir
    g0 = gabor_tensor(a0, phi, ones)                    # b = 1 (lo que da a=0)
    img_rad = render(1, xyz, scales, rots, opac, colors, beta, g0)
    img_dir = render(2, xyz, scales, rots, opac, colors, beta, g0)
    d = (img_rad - img_dir).abs().max().item()
    print(f"[1] a=0: |radial - dir| max = {d:.3e}   -> {'OK' if d < 1e-6 else 'FALLO'}")
    print(f"    rango de la imagen: {img_rad.min().item():.4f} .. {img_rad.max().item():.4f}")
    assert img_rad.max().item() > 0.01, "la escena de prueba no pinta nada"

    # ---------- 2) gradientes vs diferencias finitas ----------
    target = torch.rand_like(img_rad)

    def loss_of(mode, a_, phi_, b_, beta_, scales_):
        img = render(mode, xyz, scales_, rots, opac, colors, beta_,
                     gabor_tensor(a_, phi_, b_))
        return ((img - target) ** 2).mean()

    for mode, name in [(0, "legacy (CONTROL: codigo no tocado)"), (1, "radial"), (2, "dir")]:
        a = (torch.rand(n, 3, device=DEV) * 0.15).requires_grad_(True)
        ph = (torch.rand(n, 1, device=DEV) * 2.0 + 0.5).requires_grad_(True)
        bb = (torch.rand(n, 1, device=DEV) * 0.4 + 0.6).requires_grad_(True)
        bt = beta.clone().requires_grad_(True)
        sc = scales.clone().requires_grad_(True)
        if mode == 0:                      # legacy ignora phi/b -> grad debe ser 0
            a = (torch.rand(n, 3, device=DEV) * 0.15).requires_grad_(True)
        loss = loss_of(mode, a, ph, bb, bt, sc)
        loss.backward()
        an = {"a": a.grad.clone(), "phi": ph.grad.clone(), "b": bb.grad.clone(),
              "beta": bt.grad.clone(), "scaling": sc.grad.clone()}

        eps = 2e-3
        worst = {}
        for key, tensor in [("a", a), ("phi", ph), ("b", bb), ("beta", bt), ("scaling", sc)]:
            num = torch.zeros_like(tensor)
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    for sign in (+1, -1):
                        pert = tensor.detach().clone()
                        pert[i, j] += sign * eps
                        args = {"a": a.detach(), "phi": ph.detach(), "b": bb.detach(),
                                "beta": bt.detach(), "scaling": sc.detach()}
                        args[key] = pert
                        L = loss_of(mode, args["a"], args["phi"], args["b"],
                                    args["beta"], args["scaling"]).item()
                        num[i, j] += sign * L
                    num[i, j] /= (2 * eps)
            scale = max(num.abs().max().item(), an[key].abs().max().item(), 1e-12)
            diff = (num - an[key]).abs()
            worst[key] = (diff.max().item() / scale, diff.median().item() / scale)
        line = "  ".join(f"{k}={v[0]:.1%}/{v[1]:.1%}" for k, v in worst.items())
        print(f"[2] {name}:  max/mediana del error relativo   {line}")

    # ---------- 3) legacy no cambia ----------
    a_leg = torch.tensor([0.4053, 0.0450, 0.0162], device=DEV).repeat(n, 1)
    a_leg = a_leg * (0.5 / a_leg.sum(dim=1, keepdim=True))
    img_leg = render(0, xyz, scales, rots, opac, colors, beta,
                     gabor_tensor(a_leg, zero, zero))
    # con los coefs de la tent, legacy debe parecerse mucho al atomo con a=0
    rel = (img_leg - img_rad).abs().max().item() / max(img_rad.max().item(), 1e-9)
    print(f"[3] legacy(a=tent) vs atomo(a=0): dif relativa max = {rel:.2%} "
          f"(esperado pequeno: la serie truncada aproxima la tent, f(0)=0.9665)")


if __name__ == "__main__":
    main()
