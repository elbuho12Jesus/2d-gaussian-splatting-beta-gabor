#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
import math
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        # Spherical Betas (oficial beta_model): softplus en el rgb del lóbulo
        # (≥0, los lóbulos solo AÑADEN luz sobre el color base SH); θ/φ/b crudos.
        def sb_params_activation(sb_params):
            softplus_sb_params = F.softplus(sb_params[..., :3], beta=math.log(2) * 10)
            return torch.cat([softplus_sb_params, sb_params[..., 3:]], dim=-1)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.sb_params_activation = sb_params_activation


    def __init__(self, sh_degree : int, sb_number : int = 0):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        # Nº de lóbulos Spherical Beta por splat (0 = color solo SH).
        # _sb_params siempre existe con shape (N, sb_number, 6) — con sb_number=0
        # es (N,0,6) y todas las rutas (optimizer, prune, cat, ply) son no-op.
        self.sb_number = sb_number
        self._sb_params = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._beta = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        # ✅ DBS: contador de opacidad baja sostenida
        self.low_opacity_counter = torch.empty(0)
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # Factor del techo de escala (clamp de get_scaling): s <= factor * extent.
        # Default 0.1 = convención 3DGS/2DGS (comportamiento histórico bit-exacto).
        # Configurable por env var para A/B del clamp de los gigantes del fondo
        # (docs/clamp_escala_gigantes_fondo.html): bajarlo fuerza surfels más chicos
        # (más cobertura, riesgo de huecos), subirlo los deja crecer (riesgo velo/OOM).
        # Se lee aquí (no en get_scaling, que es hot-path) → mismo valor en train y
        # render mientras la env var esté exportada en el shell (como el run script).
        _scf = os.environ.get("SCALE_CLAMP_FACTOR", "").strip()
        self.scale_clamp_factor = float(_scf) if _scf else 0.1
        # Origen del valor: delata el fallo silencioso de run65 (script titulado
        # "small clamp" que NUNCA exportó la env var → corrió con el default 0.1).
        self.scale_clamp_source = "env SCALE_CLAMP_FACTOR" if _scf else "DEFAULT (env var NO exportada)"
        print("[CLAMP] scale_clamp_factor = {:.4f}  <- {}".format(
            self.scale_clamp_factor, self.scale_clamp_source))
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._sb_params,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._sb_params,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        # Techo de escala atado a la extensión de la escena (convención 3DGS/2DGS:
        # un splat > 0.1*extent es degenerado). Sin esto, un surfel puede crecer sin
        # límite, cubrir miles de tiles y desbordar el buffer de binning del rasterizer
        # (OOM con asignaciones absurdas tipo "1050571 GiB"). El clamp recorta el grad
        # por encima del tope, así el optimizer deja de inflarlos.
        s = self.scaling_activation(self._scaling)
        if self.spatial_lr_scale > 0:
            s = s.clamp(max=self.scale_clamp_factor * self.spatial_lr_scale)
        return s
    
    @torch.no_grad()
    def clamp_report(self, iteration):
        """Verifica que el clamp de escala se esté APLICANDO de verdad.

        Compara la escala CRUDA (activación sin clamp) contra el techo. Si el clamp
        funciona: s_max(post) == techo exacto y %topados > 0 en cuanto haya gigantes.
        Si %topados == 0 el clamp es un no-op (nadie llega al techo) y cualquier A/B
        sobre SCALE_CLAMP_FACTOR NO concluye nada — el caso que hay que detectar.
        """
        if self.spatial_lr_scale <= 0:
            print("\n[ITER {}] [CLAMP] INACTIVO (spatial_lr_scale=0)".format(iteration))
            return
        ceil = self.scale_clamp_factor * self.spatial_lr_scale
        raw = self.scaling_activation(self._scaling)          # SIN clamp
        post = raw.clamp(max=ceil)                            # lo que ve el rasterizer
        n = raw.numel()
        topped = int((raw >= ceil - 1e-9).sum().item())       # componentes en el techo
        print("\n[ITER {}] [CLAMP] factor={:.4f} techo={:.6f} | topados: {}/{} ({:.3f}%) | "
              "s_raw max/mean={:.6f}/{:.6f} -> s_post max/mean={:.6f}/{:.6f} | recorte_max={:.6f}".format(
                  iteration, self.scale_clamp_factor, ceil, topped, n, 100.0 * topped / max(n, 1),
                  raw.max().item(), raw.mean().item(),
                  post.max().item(), post.mean().item(),
                  max(raw.max().item() - ceil, 0.0)))

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)    
    
    @property
    def get_beta(self):
        # min -4.0 (igualado con el clamp de train.py:399; el -6.0 previo era holgura
        # muerta: train.py proyecta _beta a [-4,2] cada iter, nunca baja de -4).
        # max 2.7081 (run73): beta_techo 4*e^2.7081 = 60 (viejo 4*e^2 = 29.556).
        # Los DOS clamps deben moverse juntos o el forward recorta a 29.556 (NO-OP).
        b = self._beta.clamp(min=-4.0, max=2.0)
        return (4.0 * torch.exp(b)).contiguous()

    @property
    def get_sb_params(self):
        # (N, sb_number, 6) activado: rgb ≥ 0 (softplus), θ/φ/b crudos
        return self.sb_params_activation(self._sb_params)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        # Techo EFECTIVO (ya se conoce extent): s <= factor * extent. Si extent es 0
        # el clamp de get_scaling queda inactivo (guard `spatial_lr_scale > 0`).
        print("[CLAMP] extent(spatial_lr_scale) = {:.4f} | techo efectivo s_max = {:.4f} x {:.4f} = {:.6f} | activo = {}".format(
            spatial_lr_scale, self.scale_clamp_factor, spatial_lr_scale,
            self.scale_clamp_factor * spatial_lr_scale, spatial_lr_scale > 0))
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # Spherical Betas (oficial): [r,g,b,θ,φ,b] por lóbulo, rgb/b a 0 y ejes
        # (θ,φ) uniformes en la esfera para que cada lóbulo arranque mirando a una
        # dirección distinta. Con sb_number=0 queda (N,0,6) = desactivado.
        sb_params = torch.zeros((fused_point_cloud.shape[0], self.sb_number, 6), device="cuda")
        if self.sb_number > 0:
            sb_params[:, :, 3] = torch.pi * torch.rand(fused_point_cloud.shape[0], self.sb_number)      # θ ∈ [0, π]
            sb_params[:, :, 4] = 2 * torch.pi * torch.rand(fused_point_cloud.shape[0], self.sb_number)  # φ ∈ [0, 2π]

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._sb_params = nn.Parameter(sb_params.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))       
        betas = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        self._beta = nn.Parameter(betas.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.low_opacity_counter = torch.zeros((self.get_xyz.shape[0],), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.beta_densify_threshold = getattr(training_args, "beta_densify_threshold", 0.0)
        self.beta_densify_mode = getattr(training_args, "beta_densify_mode", "split_wide")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._sb_params], 'lr': getattr(training_args, "sb_params_lr", 0.0025), "name": "sb_params"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._beta], 'lr': training_args.beta_lr, "name": "beta"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        # Spherical Betas: layout channel-major (N,6,M) aplanado, como el oficial
        for i in range(self._sb_params.shape[1]*self._sb_params.shape[2]):
            l.append('sb_params_{}'.format(i))
        l.append('opacity')
        l.append('beta')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sb_params = self._sb_params.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        beta = self._beta.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, sb_params, opacities, beta, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, iteration=None):
        if os.environ.get("DEBUG_DENSIFY", "1") != "0":
            cur = self.get_opacity.squeeze(-1)
            n_above = int((cur > 0.01).sum())
            print(f"[RESET iter={iteration}] reset_opacity -> 0.01 | bajados (opac>0.01)={n_above}/{cur.shape[0]} "
                  f"| opac<0.005 antes={int((cur < 0.005).sum())}", flush=True)
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # Spherical Betas: ply guarda (N,6,M) channel-major aplanado (layout oficial).
        # Si el ply es antiguo (sin campos sb_params_*) y sb_number>0, se inicializa
        # fresco con ejes aleatorios (rgb=0 → contribución nula hasta entrenar).
        sb_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sb_params_")]
        sb_names = sorted(sb_names, key=lambda x: int(x.split('_')[-1]))
        if len(sb_names) > 0:
            self.sb_number = len(sb_names) // 6
            sb_params = np.zeros((xyz.shape[0], len(sb_names)))
            for idx, attr_name in enumerate(sb_names):
                sb_params[:, idx] = np.asarray(plydata.elements[0][attr_name])
            sb_params = sb_params.reshape((sb_params.shape[0], 6, self.sb_number))
            sb_t = torch.tensor(sb_params, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        else:
            sb_t = torch.zeros((xyz.shape[0], self.sb_number, 6), dtype=torch.float, device="cuda")
            if self.sb_number > 0:
                sb_t[:, :, 3] = torch.pi * torch.rand(xyz.shape[0], self.sb_number, device="cuda")
                sb_t[:, :, 4] = 2 * torch.pi * torch.rand(xyz.shape[0], self.sb_number, device="cuda")

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._sb_params = nn.Parameter(sb_t.requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        # ✅ beta: LEER el valor entrenado del ply (no reinicializar). El bug previo
        # ponía beta=1.0 a todos al cargar → descartaba el beta entrenado (raw mean
        # ~-3.45) → render.py/metrics.py daban PSNR ~11 ("colapso a negro") aunque el
        # modelo era bueno (~20 en el eval interno del entrenamiento). Fallback a 1.0
        # solo si el ply es antiguo y no tiene el campo 'beta' (mismo patrón que sb_params).
        prop_names = [p.name for p in plydata.elements[0].properties]
        if "beta" in prop_names:
            beta = np.asarray(plydata.elements[0]["beta"])[..., np.newaxis]
        else:
            beta = np.ones((xyz.shape[0], 1), dtype=np.float32)
        self._beta = nn.Parameter(torch.tensor(beta, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._sb_params = optimizable_tensors["sb_params"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._beta = optimizable_tensors["beta"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.low_opacity_counter = self.low_opacity_counter[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_beta, new_scaling, new_rotation, new_sb_params=None):
        if new_sb_params is None:
            new_sb_params = torch.zeros((new_xyz.shape[0], self.sb_number, 6), device=new_xyz.device)
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "sb_params": new_sb_params,
        "opacity": new_opacities,
        "beta": new_beta,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._sb_params = optimizable_tensors["sb_params"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # low_opacity_counter: PRESERVAR el conteo de los splats existentes (los nuevos
        # arrancan en 0). Reiniciarlo a ceros aquí rompía el conteo sostenido en el
        # camino MCMC, porque add_new_gs llama a este postfix CADA paso → el contador
        # nunca pasaba de 1 (mismo síndrome de código muerto que el opacity_reset en el
        # clásico). El concat mantiene el historial; relocate_gs/prune_points ya lo
        # mantienen alineado. Los otros acumuladores SÍ se reinician (es correcto: la
        # acumulación de gradiente arranca de cero tras cambiar el set de splats).
        n_new = new_xyz.shape[0]
        n_total = self.get_xyz.shape[0]
        if self.low_opacity_counter.shape[0] == n_total - n_new:
            self.low_opacity_counter = torch.cat(
                [self.low_opacity_counter, torch.zeros(n_new, device="cuda")])
        else:
            self.low_opacity_counter = torch.zeros((n_total,), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # ---------- FILTRO OPCIONAL POR BETA (insertar aquí) ----------
        # Requiere que hayas añadido `self.beta_densify_threshold` (por ejemplo en arguments.py)
        # y que la hayas inicializado en la instancia (p.ej. en training_setup).
        if hasattr(self, "beta_densify_threshold") and self.beta_densify_threshold > 0.0:
            beta_vals = self.get_beta.squeeze()
            if beta_vals.shape[0] != n_init_points:
                beta_vals = beta_vals[:n_init_points]
            mode = getattr(self, "beta_densify_mode", "split_wide")
            if mode == "split_wide":
                selected_pts_mask = selected_pts_mask & (beta_vals <= self.beta_densify_threshold)
            else:
                selected_pts_mask = selected_pts_mask & (beta_vals >= self.beta_densify_threshold)
        # safety: si no hay puntos seleccionados, salir
        if selected_pts_mask.sum() == 0:
            return
        # ---------- fin filtro por beta ----------

        # ===============================
        # ✅ Deterministic DBS split
        # ===============================

        xyz = self.get_xyz[selected_pts_mask]
        scales = self.get_scaling[selected_pts_mask]
        rots = build_rotation(self._rotation[selected_pts_mask])

        # elegir eje dominante (mayor escala)
        mask = (scales[:, 0] > scales[:, 1]).float().unsqueeze(1)

        # eje principal en espacio local
        v1_local = torch.cat([
            mask,                  # eje x si s_x > s_y
            1 - mask,              # eje y si no
            torch.zeros_like(mask) # sin componente normal
        ], dim=1)

        # llevar a espacio mundo
        v1_world = torch.bmm(rots, v1_local.unsqueeze(-1)).squeeze(-1)

        # magnitud del split (proporcional al tamaño)
        delta = 0.5 * torch.max(scales, dim=1).values.unsqueeze(1)

        # crear dos hijos
        new_xyz = torch.cat([
            xyz + delta * v1_world,
            xyz - delta * v1_world
        ], dim=0)
        
        new_scaling = self.scaling_inverse_activation(
            (self.get_scaling[selected_pts_mask] / (0.8 * N)).repeat(N, 1)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)        
        # ✅ DBS-style split: alpha is evenly divided
        alpha = self.get_opacity[selected_pts_mask]
        alpha_new = alpha / N
        new_opacity = self.inverse_opacity_activation(alpha_new).repeat(N, 1)

        # FIX 2026-07-20 (trinquete de beta, run67): ANTES restaba math.log(N), lo que
        # en el espacio activado (beta = 4*exp(_beta)) equivale a DIVIDIR beta entre N
        # en CADA generacion de split -> tras 6 splits el hijo queda clavado en el suelo
        # del clamp (_beta=-4 -> beta=0.0733) y su kernel degenera a CAJA. La analogia
        # con escala (/0.8N) y opacidad (/N) era erronea: esas son magnitudes EXTENSIVAS
        # y beta es el parametro de FORMA del kernel, adimensional -> no se reparte.
        # Ahora hereda beta tal cual, consistente con densify_and_clone (599),
        # relocate_gs (771) y add_new_gs (849), las otras 3 rutas de creacion.
        # Detalle: docs/beta_trinquete_split_clasico.html
        new_beta = self._beta[selected_pts_mask].repeat(N, 1)
        new_sb_params = self._sb_params[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_beta, new_scaling, new_rotation, new_sb_params)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        # ✅ DBS-style clone: preserve transmittance
        alpha = self.get_opacity[selected_pts_mask]      # (0,1)
        K = 1  # clone crea 1 copia adicional por punto
        alpha_new = alpha / (K + 1)

        new_opacities = self.inverse_opacity_activation(alpha_new)

        new_beta = self._beta[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_sb_params = self._sb_params[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_beta, new_scaling, new_rotation, new_sb_params)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration=None, prune_sustain=0):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        _dbg = os.environ.get("DEBUG_DENSIFY", "1") != "0"
        if _dbg:
            # Gradiente de posición (viewspace) acumulado por splat = "error" que dirige
            # clone/split. Solo cuentan los splats vistos al menos una vez (denom>0).
            gnorm = grads.squeeze(-1)
            seen = self.denom.squeeze(-1) > 0
            gv = gnorm[seen]
            n_total = self.get_xyz.shape[0]
            n_cand = int((gv >= max_grad).sum()) if gv.numel() else 0
            if gv.numel():
                qs = torch.quantile(gv, torch.tensor([0.5, 0.9, 0.99], device=gv.device))
                gstats = (f"mean={gv.mean().item():.2e} med={qs[0].item():.2e} "
                          f"p90={qs[1].item():.2e} p99={qs[2].item():.2e} max={gv.max().item():.2e}")
            else:
                gstats = "sin grads"
            print(f"[DENSIFY iter={iteration}] N={n_total} vistos={int(seen.sum())} "
                  f"grad_pos(thr={max_grad:.1e}): {gstats} | >=thr={n_cand} "
                  f"({100.0*n_cand/max(n_total,1):.2f}%)", flush=True)

        n0 = self.get_xyz.shape[0]
        self.densify_and_clone(grads, max_grad, extent)
        n1 = self.get_xyz.shape[0]
        self.densify_and_split(grads, max_grad, extent)
        n2 = self.get_xyz.shape[0]

        # ============================================================
        # PRUNE por opacidad. Dos modos (prune_sustain controla cuál):
        #  - prune_sustain=0 (default): PRUNE INMEDIATO 2DGS/3DGS original (run15) →
        #    opacidad < cull AHORA se poda en el acto.
        #  - prune_sustain=N (>0): PRUNE SOSTENIDO → el splat debe llevar N pasos de
        #    densify CONSECUTIVOS bajo el cull (low_opacity_counter > N) antes de podarse,
        #    dándole tiempo a recuperar opacidad ("asentarse") por gradiente. ADVERTENCIA:
        #    si opacity_reset cae cada M densifies con M<=N, el reset sube todo a 0.01>cull
        #    → counter se borra → criterio jamás se cumple = CÓDIGO MUERTO (caso run14,
        #    N=50 con M=30). Para que N=30 funcione, opacity_reset debe estar OFF o con
        #    intervalo >> N·densification_interval. Ver low_opacity_counter_y_reset.html.
        # En ambos modos se suma el prune por tamaño (screen/world), siempre inmediato.
        # ============================================================
        low_now = (self.get_opacity < min_opacity).squeeze()
        if prune_sustain > 0:
            if self.low_opacity_counter.shape[0] != low_now.shape[0]:
                # Realineación defensiva (no debería pasar: postfix/prune lo mantienen).
                self.low_opacity_counter = torch.zeros_like(low_now, dtype=torch.float, device="cuda")
            self.low_opacity_counter[low_now] += 1
            self.low_opacity_counter[~low_now] = 0
            prune_alpha_mask = self.low_opacity_counter > prune_sustain
        else:
            prune_alpha_mask = low_now
        prune_mask = prune_alpha_mask
        big_points_vs = None
        big_points_ws = None
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if _dbg:
            n_before = self.get_xyz.shape[0]
            n_alpha = int(prune_alpha_mask.sum())
            n_screen = int(big_points_vs.sum()) if big_points_vs is not None else 0
            n_world = int(big_points_ws.sum()) if big_points_ws is not None else 0
            if prune_sustain > 0:
                alpha_lbl = (f"sostenido(>{prune_sustain})={n_alpha} "
                             f"[low_now={int(low_now.sum())} cnt_max={int(self.low_opacity_counter.max())}]")
            else:
                alpha_lbl = f"opac<{min_opacity:g}={n_alpha}"
            print(f"[DENSIFY iter={iteration}] +clone={n1-n0} +split={n2-n1} "
                  f"(sel={(n2-n1)}) | PRUNE total={int(prune_mask.sum())} "
                  f"[{alpha_lbl}, screen={n_screen}, world={n_world}] | "
                  f"N:{n_before}->{n_before-int(prune_mask.sum())}", flush=True)

        # ---- aplicar pruning ----
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def _reset_optimizer_state(self, inds):
        """Pone exp_avg / exp_avg_sq a 0 para los índices dados (in-place)."""
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is None:
                continue
            stored_state["exp_avg"][inds] = 0
            stored_state["exp_avg_sq"][inds] = 0

    def _error_signal(self, clip=10.0):
        """Error de reconstrucción por splat ∝ gradiente de viewspace acumulado.
        Es el proxy clásico de 3DGS para "splat en zona sub-reconstruida": se
        acumula en add_densification_stats entre densificaciones y densification_postfix
        lo resetea al añadir splats. Devuelve un tensor [N] normalizado por la media
        de los splats con señal (media ≈ 1) y clampeado a `clip` para que ningún
        outlier domine el muestreo multinomial. Splats sin señal (no visibles en el
        intervalo) reciben 0 → caen a muestreo por opacidad pura.
        """
        if self.denom.numel() == 0 or self.xyz_gradient_accum.numel() == 0:
            return torch.zeros(self._xyz.shape[0], device=self._xyz.device)
        denom = self.denom.clamp(min=1.0)
        grad = (self.xyz_gradient_accum / denom).squeeze(-1)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        pos = grad > 0
        if not pos.any():
            return torch.zeros_like(grad)
        mean = grad[pos].mean()
        if not torch.isfinite(mean) or mean <= 0:
            return torch.zeros_like(grad)
        return (grad / mean).clamp(max=clip)

    def relocate_gs(self, dead_mask, error_weight=0.0):
        """MCMC-style relocation (alineado con Beta Splatting oficial).
        Copia splats vivos en los slots muertos. Sin jitter posicional — la
        perturbación viene del paso de ruido posterior en train.py. Aplica la
        regla de conservación de transmittance: si un src se elige 'ratio' veces habrá
        ratio+1 instancias (copias + original), todas con new_alpha = 1-(1-src_alpha)^(1/(ratio+1))
        para conservar la transmitancia total (ver bloque de código abajo). NOTA: NO es src/2;
        esa era la versión vieja, solo correcta si ratio==1.
        Con error_weight>0, el muestreo de fuentes se sesga hacia splats de alto
        error de reconstrucción (no solo alta opacidad) para cubrir zonas mal resueltas.
        """
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return
        alive_mask = ~dead_mask
        dead_idx = dead_mask.nonzero(as_tuple=True)[0]
        alive_idx = alive_mask.nonzero(as_tuple=True)[0]
        if alive_idx.shape[0] == 0:
            return

        alive_op = self.get_opacity.squeeze()[alive_idx]
        if error_weight > 0.0:
            err = self._error_signal()
            alive_op = alive_op * (1.0 + error_weight * err[alive_idx])
        alive_op = torch.nan_to_num(alive_op, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-6)
        total = alive_op.sum()
        if not torch.isfinite(total) or total <= 0:
            print(f"[relocate_gs] WARNING: pesos no válidos (sum={total.item()}); se omite relocate este paso")
            return
        weights = alive_op / total
        src_idx = alive_idx[torch.multinomial(weights, n_dead, replacement=True)]

        with torch.no_grad():
            # Regla de preservación de distribución (oficial beta_model._update_params):
            # si un src se elige 'ratio' veces por multinomial, habrá ratio+1 instancias
            # (las copias muertas reubicadas + el propio src). Cada una recibe
            # new_alpha = 1-(1-alpha)^(1/(ratio+1)) para conservar la transmitancia total.
            # (Antes: alpha/2, que solo es correcto si ratio==1 y sobre-infla la opacidad
            # de los src populares -> sesga capacidad al foreground y desestabiliza.)
            ratio = torch.bincount(src_idx)[src_idx].to(self._xyz.dtype).unsqueeze(-1)
            src_alpha = self.get_opacity[src_idx].clamp(min=2e-3, max=1.0 - 1e-3)
            new_alpha = (1.0 - torch.pow(1.0 - src_alpha, 1.0 / (ratio + 1.0))).clamp(min=5e-3, max=1.0 - 1e-3)
            new_opacity_raw = self.inverse_opacity_activation(new_alpha)

            self._xyz[dead_idx] = self._xyz[src_idx]
            self._features_dc[dead_idx] = self._features_dc[src_idx]
            self._features_rest[dead_idx] = self._features_rest[src_idx]
            self._sb_params[dead_idx] = self._sb_params[src_idx]
            self._scaling[dead_idx] = self._scaling[src_idx]
            self._rotation[dead_idx] = self._rotation[src_idx]
            self._beta[dead_idx] = self._beta[src_idx]
            self._opacity[dead_idx] = new_opacity_raw
            self._opacity[src_idx] = new_opacity_raw

            self.low_opacity_counter[dead_idx] = 0
            self.xyz_gradient_accum[dead_idx] = 0.0
            self.denom[dead_idx] = 0.0
            self.max_radii2D[dead_idx] = 0.0

        reset_idx = torch.cat([dead_idx, src_idx]).unique()
        self._reset_optimizer_state(reset_idx)

    def add_new_gs(self, cap_max, error_weight=0.0, jitter_scale=0.0):
        """Crece hasta cap_max copiando splats vivos.
        Con error_weight>0, el muestreo de fuentes se sesga hacia splats de alto
        error de reconstrucción para sembrar nuevos splats cerca de zonas mal resueltas.
        Con jitter_scale>0, los clones se desplazan a lo largo del eje in-plane
        DOMINANTE del surfel (marco propio, como densify_and_split), signo ±
        aleatorio, con magnitud ∝ escala_dominante × jitter_scale × err_src — el clon
        queda SOBRE el plano del surfel (no se va por la normal) y un src en el BORDE
        de un hueco puede sembrar el INTERIOR del hueco, no solo más borde.
        El jitter solo se aplica si error_weight>0 (necesita la señal de error).
        """
        cur = self._xyz.shape[0]
        target = min(int(cap_max), int(math.ceil(1.05 * cur)))
        k = target - cur
        if k <= 0:
            return

        use_error = error_weight > 0.0
        err = self._error_signal() if use_error else None

        probs = self.get_opacity.squeeze()
        if use_error:
            probs = probs * (1.0 + error_weight * err)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-6)
        total = probs.sum()
        if not torch.isfinite(total) or total <= 0:
            print(f"[add_new_gs] WARNING: pesos no válidos (sum={total.item()}); se omite add_new este paso")
            return
        probs = probs / total
        src_idx = torch.multinomial(probs, k, replacement=True)

        with torch.no_grad():
            # Preservación de distribución (oficial beta_model._update_params): cada src
            # elegido 'ratio' veces genera ratio clones + queda el original = ratio+1
            # instancias, cada una con new_alpha = 1-(1-alpha)^(1/(ratio+1)).
            # (Antes: alpha/2 -> sobre-densidad en zonas opacas si el src se clona >1 vez.)
            ratio = torch.bincount(src_idx)[src_idx].to(self._xyz.dtype).unsqueeze(-1)
            src_alpha = self.get_opacity[src_idx].clamp(min=2e-3, max=1.0 - 1e-3)
            new_alpha = (1.0 - torch.pow(1.0 - src_alpha, 1.0 / (ratio + 1.0))).clamp(min=5e-3, max=1.0 - 1e-3)
            new_opacity_raw = self.inverse_opacity_activation(new_alpha)

            new_xyz = self._xyz[src_idx].detach().clone()
            if jitter_scale > 0.0 and use_error:
                # Desplazamiento en el MARCO PROPIO del surfel (como densify_and_split):
                # a lo largo del eje IN-PLANE DOMINANTE (el de mayor escala), llevado a
                # mundo con la rotación del src. Así el clon queda SOBRE el plano del
                # surfel (nunca se va por la normal), a diferencia de la dirección 3D
                # libre anterior que sacaba clones fuera del plano. Magnitud = escala del
                # eje dominante × jitter_scale × err_src (alto error desplaza más →
                # siembra huecos; bajo error → clone casi in situ). Signo ± aleatorio
                # (split crea los dos hijos ±; aquí cada clon toma un signo al azar).
                src_scales = self.get_scaling[src_idx]                              # (k, 2)
                rots = build_rotation(self._rotation[src_idx])                      # (k, 3, 3)
                dom = (src_scales[:, 0] > src_scales[:, 1]).float().unsqueeze(1)    # eje x si s_x>s_y
                v1_local = torch.cat([dom, 1.0 - dom, torch.zeros_like(dom)], dim=1)  # eje dom. local (normal=0)
                v1_world = torch.bmm(rots, v1_local.unsqueeze(-1)).squeeze(-1)      # a mundo (k, 3)
                dom_scale = torch.max(src_scales, dim=1).values.unsqueeze(1)        # escala del eje dominante
                sign = torch.randn(new_xyz.shape[0], 1, device=new_xyz.device).sign()
                sign[sign == 0] = 1.0
                err_src = err[src_idx].unsqueeze(-1)
                new_xyz = new_xyz + sign * v1_world * dom_scale * (jitter_scale * err_src)
            new_features_dc = self._features_dc[src_idx].detach().clone()
            new_features_rest = self._features_rest[src_idx].detach().clone()
            new_sb_params = self._sb_params[src_idx].detach().clone()
            new_scaling = self._scaling[src_idx].detach().clone()
            new_rotation = self._rotation[src_idx].detach().clone()
            new_beta = self._beta[src_idx].detach().clone()
            new_opacity = new_opacity_raw.detach().clone()

            # Reducir opacidad de los srcs ANTES del postfix (que reasigna _opacity).
            self._opacity[src_idx] = new_opacity_raw

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_beta, new_scaling, new_rotation, new_sb_params,
        )
        self._reset_optimizer_state(src_idx.unique())

    def sanitize_parameters(self, iteration=None):
        """Detecta NaN/Inf en parámetros y los repara in-place.
        Devuelve el número de splats saneados. Marca los splats afectados como
        casi-muertos (opacity ≈ 1e-4) para que el siguiente relocate los recicle.
        Resetea momentum/varianza del optimizer para esos índices.
        """
        with torch.no_grad():
            def bad_rows(t):
                if t.dim() == 1:
                    return ~torch.isfinite(t)
                return ~torch.isfinite(t.view(t.shape[0], -1)).all(dim=1)

            mask = bad_rows(self._xyz)
            mask = mask | bad_rows(self._opacity)
            mask = mask | bad_rows(self._scaling)
            mask = mask | bad_rows(self._rotation)
            mask = mask | bad_rows(self._features_dc)
            mask = mask | bad_rows(self._features_rest)
            mask = mask | bad_rows(self._sb_params)
            mask = mask | bad_rows(self._beta)
            n_bad = int(mask.sum().item())
            if n_bad == 0:
                return 0

            tag = f"[Iter {iteration}] " if iteration is not None else ""
            print(f"{tag}WARNING: detectados {n_bad} splats con NaN/Inf — saneando")

            dev = self._xyz.device
            self._xyz.data[mask] = 0.0
            # opacity ≈ 1e-4 ⇒ caerá bajo opacity_cull (0.005) ⇒ relocate la próxima vez
            tiny_alpha = torch.tensor(1e-4, device=dev)
            self._opacity.data[mask] = self.inverse_opacity_activation(tiny_alpha)
            # scale en log-space; exp(-5) ≈ 6.7e-3
            self._scaling.data[mask] = -5.0
            # quaternion identidad (1,0,0,0)
            self._rotation.data[mask] = 0.0
            self._rotation.data[mask, 0] = 1.0
            self._features_dc.data[mask] = 0.0
            self._features_rest.data[mask] = 0.0
            self._sb_params.data[mask] = 0.0
            self._beta.data[mask] = 0.0
            # buffers densificación
            if self.xyz_gradient_accum.shape[0] == mask.shape[0]:
                self.xyz_gradient_accum[mask] = 0.0
                self.denom[mask] = 0.0
                self.max_radii2D[mask] = 0.0
            if self.low_opacity_counter.shape[0] == mask.shape[0]:
                self.low_opacity_counter[mask] = 0

            self._reset_optimizer_state(mask.nonzero(as_tuple=True)[0])
            return n_bad

    def prune_nan_splats(self, iteration=None):
        """Elimina definitivamente los splats con NaN/Inf en cualquier parámetro.
        A diferencia de sanitize_parameters (que los recicla), prune_nan_splats los
        borra del modelo. Usar antes del save final para que el PLY no tenga NaN.
        """
        with torch.no_grad():
            def bad_rows(t):
                if t.dim() == 1:
                    return ~torch.isfinite(t)
                return ~torch.isfinite(t.view(t.shape[0], -1)).all(dim=1)

            mask = bad_rows(self._xyz)
            mask = mask | bad_rows(self._opacity)
            mask = mask | bad_rows(self._scaling)
            mask = mask | bad_rows(self._rotation)
            mask = mask | bad_rows(self._features_dc)
            mask = mask | bad_rows(self._features_rest)
            mask = mask | bad_rows(self._sb_params)
            mask = mask | bad_rows(self._beta)
            n_bad = int(mask.sum().item())
            if n_bad == 0:
                return 0
            tag = f"[Iter {iteration}] " if iteration is not None else ""
            print(f"{tag}prune_nan_splats: eliminando {n_bad} splats con NaN/Inf")
        self.prune_points(mask)
        return n_bad