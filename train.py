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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_rotation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def print_memory(tag):
    print(f"[{tag}] allocated={torch.cuda.memory_allocated()/1e9:.3f}GB reserved={torch.cuda.memory_reserved()/1e9:.3f}GB max_reserved={torch.cuda.max_memory_reserved()/1e9:.3f}GB")

# Instrumentación de debug, activada por variables de entorno (0/ausente = off):
#   DEBUG_MEM=N    -> imprime memoria cada N iters, en cada punto del step (post_render/
#                     post_backward/post_step) con el PICO por iteración reseteado al inicio.
#   DEBUG_NOISE=N  -> imprime estadísticas del ruido posicional cada N aplicaciones de ruido.
DEBUG_MEM = int(os.environ.get("DEBUG_MEM", "0"))
DEBUG_NOISE = int(os.environ.get("DEBUG_NOISE", "0"))

def mem_probe(tag, iteration, npts=None):
    """Una línea compacta con alloc/reserved actuales, el pico de la iteración y la
    memoria LIBRE real del device (delata procesos zombie ocupando VRAM)."""
    if not DEBUG_MEM or iteration % DEBUG_MEM != 0:
        return
    a  = torch.cuda.memory_allocated() / 1e9
    r  = torch.cuda.memory_reserved() / 1e9
    pa = torch.cuda.max_memory_allocated() / 1e9
    pr = torch.cuda.max_memory_reserved() / 1e9
    free, total = torch.cuda.mem_get_info()
    extra = f" npts={npts}" if npts is not None else ""
    print(f"[MEM it{iteration} {tag}] alloc={a:.2f} reserved={r:.2f} "
          f"peak_alloc={pa:.2f} peak_reserved={pr:.2f} "
          f"dev_free={free/1e9:.2f}/{total/1e9:.2f}GB{extra}")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, getattr(dataset, "sb_number", 0))
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # --- Resolución del algoritmo de densificación (UNA sola fuente de verdad) ---
    # Interfaz preferida: --densify_mode {mcmc,classic}. Alias retrocompatible:
    # --classic_densify (si se pasa, FUERZA clásico). Las dos rutas están aisladas
    # más abajo (if classic / elif mcmc / else): cambiar de modo no afecta a la otra.
    densify_mode = str(getattr(opt, "densify_mode", "mcmc")).lower()
    if densify_mode not in ("mcmc", "classic"):
        raise ValueError(f"--densify_mode debe ser 'mcmc' o 'classic', recibido: {densify_mode!r}")
    use_classic_densify = (densify_mode == "classic") or bool(getattr(opt, "classic_densify", False))
    print(f"[DENSIFY] modo = {'CLASSIC (clone/split 2DGS)' if use_classic_densify else 'MCMC (relocate/add_new)'}"
          f"  (densify_mode={densify_mode}, classic_densify={bool(getattr(opt, 'classic_densify', False))})")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Centros de las cámaras de train (C,3) para el cull anti-floater: splats que
    # viven más cerca de una cámara que cualquier geometría legítima (medido en
    # flowers4: p1 de dist mínima = 0.25·extent) son floaters → reciclar vía relocate.
    cam_centers = torch.stack(
        [c.camera_center for c in scene.getTrainCameras()]).to("cuda")  # (C,3)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        # Pico de memoria POR step: lo reseteamos aquí para que peak_alloc/peak_reserved
        # midan el máximo dentro de esta iteración (delata el spike del binning del rasterizer).
        if DEBUG_MEM and iteration % DEBUG_MEM == 0:
            torch.cuda.reset_peak_memory_stats()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        mem_probe("post_render", iteration, npts=gaussians.get_xyz.shape[0])
                
        # 🔍 Debug beta (cada 100 iteraciones)
        if iteration % 5000 == 0:
            with torch.no_grad():
                beta = gaussians.get_beta
                print(
                    f"[Iter {iteration}] "
                    f"beta mean={beta.mean().item():.4f}, "
                    f"min={beta.min().item():.4f}, "
                    f"max={beta.max().item():.4f}"
                )

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]        
        if torch.isfinite(normal_error).all():
            normal_loss = lambda_normal * normal_error.mean()
        else:
            normal_loss = 0.0
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss

        # Regularizers (matching official Beta Splatting): opacity_reg + scale_reg L1.
        # Active only during densification window, as in beta-splatting/train.py.
        if opt.densify_from_iter < iteration < opt.densify_until_iter:
            total_loss = total_loss + opt.opacity_reg * gaussians.get_opacity.abs().mean()
            total_loss = total_loss + opt.scale_reg * gaussians.get_scaling.abs().mean()

        total_loss.backward()
        mem_probe("post_backward", iteration, npts=gaussians.get_xyz.shape[0])

        # ✅ DEBUG gradiente de beta (DESPUÉS del backward)
        if iteration % 10000 == 0 and gaussians._beta.grad is not None:
            print(
                "grad beta mean:",
                gaussians._beta.grad.mean().item()
            )

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # Prune definitivo de NaN antes de guardar: evita zonas negras al renderizar.
                gaussians.prune_nan_splats(iteration=iteration)
                scene.save(iteration)


            # Acumular señal de error de reconstrucción por splat (norma del gradiente
            # de viewspace). La consume el muestreo MCMC sesgado por error en
            # relocate_gs/add_new_gs. Se acumula en cada iteración del intervalo;
            # densification_postfix la resetea al añadir splats.
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # El prune clásico por tamaño en pantalla (size_threshold tras el primer
                # opacity_reset) necesita el radio 2D máximo por splat. El MCMC no lo usa.
                if use_classic_densify:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # ============================================================
            # DENSIFICACIÓN CLÁSICA 2DGS (clone/split + prune + opacity_reset).
            # Activada por --classic_densify. Sin ruido MCMC ni cull de floaters
            # (son del camino MCMC). Réplica de la orquestación del 2DGS original.
            # ============================================================
            if iteration < opt.densify_until_iter and use_classic_densify:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Sanea NaN/Inf antes de densificar (mismo guard que el MCMC).
                    gaussians.sanitize_parameters(iteration=iteration)
                    # size_threshold: prune por radio en pantalla, solo DESPUÉS del
                    # primer opacity_reset (idéntico al original 2DGS train.py).
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.opacity_cull,
                        gaussians.spatial_lr_scale, size_threshold, iteration=iteration,
                        prune_sustain=int(getattr(opt, "classic_prune_sustain", 0)))
                # opacity_reset clásico: condición simple del original (sin el gate
                # reset_cutoff del MCMC). Es SEGURO aquí porque el ruido (1−o)^100
                # está OFF en este camino → no abre ninguna compuerta de terremoto.
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity(iteration=iteration)

            # Densification (MCMC, alineado con Beta Splatting oficial).
            # Solo relocate + add_new (sin densify_and_prune 2DGS). Ruido posicional
            # se aplica únicamente en los pasos de densify para evitar la divergencia
            # numérica que se observó en Run 2 (NaN tras iter 30000).
            elif iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Defensa: sanear NaN/Inf que puedan haber entrado por gradientes
                    # explosivos o por desplazamientos extremos del ruido posicional.
                    gaussians.sanitize_parameters(iteration=iteration)
                    # dead_mask = splats a reciclar (relocate). Dos modos:
                    #  - instantáneo (mcmc_dead_sustain=0, default): opacidad <= cull AHORA.
                    #  - sostenido (>0): exige N checks consecutivos bajo el cull
                    #    (low_opacity_counter > N). Evita reubicar splats del fondo que solo
                    #    fluctúan un instante. Viable porque el camino MCMC no tiene
                    #    opacity_reset que borre el contador (ver low_opacity_counter_y_reset.html).
                    low_now = (gaussians.get_opacity <= opt.opacity_cull).squeeze(-1)
                    n_sustain = int(getattr(opt, "mcmc_dead_sustain", 0))
                    if n_sustain > 0:
                        gaussians.low_opacity_counter[low_now] += 1
                        gaussians.low_opacity_counter[~low_now] = 0
                        dead_mask = gaussians.low_opacity_counter > n_sustain
                        if DEBUG_NOISE and (iteration // opt.densification_interval) % DEBUG_NOISE == 0:
                            print(f"[DEADGATE it{iteration}] opac<=cull_ahora={int(low_now.sum())} "
                                  f"sostenido(>{n_sustain})={int(dead_mask.sum())} "
                                  f"counter_max={int(gaussians.low_opacity_counter.max())}")
                    else:
                        dead_mask = low_now

                    # Cull anti-floater: splats más cerca de un centro de cámara de
                    # train que floater_cull_dist·extent se marcan muertos → relocate
                    # los recicla a zonas de alto error (no se pierde presupuesto).
                    # Por chunks para no materializar la matriz (N,C) completa.
                    if getattr(opt, "floater_cull_dist", 0.0) > 0:
                        cull_r = opt.floater_cull_dist * gaussians.spatial_lr_scale
                        xyz = gaussians.get_xyz
                        near_cam = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
                        for i0 in range(0, xyz.shape[0], 2_000_000):
                            d = torch.cdist(xyz[i0:i0 + 2_000_000], cam_centers)  # (chunk,C)
                            near_cam[i0:i0 + 2_000_000] = d.min(dim=1).values < cull_r
                        if DEBUG_NOISE and (iteration // opt.densification_interval) % DEBUG_NOISE == 0:
                            print(f"[FLOATER it{iteration}] cull_r={cull_r:.3f} "
                                  f"n_culled={int(near_cam.sum().item())} "
                                  f"({100 * near_cam.float().mean().item():.3f}%)")
                        dead_mask = dead_mask | near_cam

                    gaussians.relocate_gs(dead_mask=dead_mask, error_weight=opt.mcmc_error_weight)
                    gaussians.add_new_gs(cap_max=opt.cap_max, error_weight=opt.mcmc_error_weight, jitter_scale=opt.mcmc_jitter_scale)

                    # Ruido posicional MCMC. Dos modos:
                    #  - isotrópico (default): randn esférico, ignora la forma del splat.
                    #  - híbrido covarianza (opt.cov_noise): anisotropía en el PLANO del
                    #    surfel (las 2 escalas) + isotrópico en la NORMAL. No usamos
                    #    build_scaling_rotation (accede a s[:,2] → IndexError con scales 2D);
                    #    construimos R con build_rotation y aplicamos un std local de 3 ejes.
                    #    MAGNITUD ∝ TAMAÑO REAL (plan #2, 2026-06-05): usamos s CRUDO (no
                    #    normalizado a media 1), así el ruido escala con el tamaño del splat
                    #    como en el oficial (ruido ∝ Σ): el fondo (splats grandes) viaja a
                    #    tapar huecos y el primer plano (splats pequeños) queda quieto. s ya
                    #    está acotado por el clamp 0.1·extent de get_scaling → sin teletransporte
                    #    de degenerados. noise_lr re-calibrado. Ver docs/ruido_isotropico_*.
                    with torch.no_grad():
                        noise_exp = float(getattr(opt, "noise_opacity_exponent", 100.0))
                        noise_lr = float(getattr(opt, "noise_lr", 5e5))
                        base_noise = torch.randn_like(gaussians._xyz)
                        noise_mult = torch.pow(1.0 - gaussians.get_opacity, noise_exp)
                        if bool(getattr(opt, "cov_noise", False)):
                            s = gaussians.get_scaling                                  # (N,2) escalas REALES del plano (ya clamp 0.1·extent)
                            pad = float(getattr(opt, "cov_noise_normal", 1.0))         # 0 = confinado al plano
                            normal_std = pad * s.mean(dim=1, keepdim=True)             # normal ∝ tasa media del plano (∝ tamaño)
                            local_std = torch.cat([s, normal_std], dim=1)             # (N,3): ∝ tamaño real [u, v, normal]
                            R = build_rotation(gaussians.get_rotation)                 # (N,3,3), col 2 = normal
                            base_noise = torch.bmm(
                                R, (local_std * base_noise).unsqueeze(-1)).squeeze(-1) # moldea y rota al mundo
                        noise = base_noise * noise_mult * noise_lr * float(xyz_lr)

                        # Debug del ruido: ¿qué splats se mueven y cuánto? Con noise_exp=100
                        # solo los de opacidad baja deberían moverse; el desplazamiento debe
                        # ser pequeño frente a la escala de escena (spatial_lr_scale).
                        if DEBUG_NOISE and (iteration // opt.densification_interval) % DEBUG_NOISE == 0:
                            op = gaussians.get_opacity.squeeze(-1)
                            disp = noise.norm(dim=1)                         # |Δxyz| por splat
                            mult = noise_mult.squeeze(-1)
                            active = mult > 1e-3                            # splats realmente perturbados
                            print(f"[NOISE it{iteration}] N={op.numel()} "
                                  f"op(mean={op.mean().item():.3f} min={op.min().item():.3f}) "
                                  f"mult(mean={mult.mean().item():.2e} max={mult.max().item():.2e}) "
                                  f"activos={int(active.sum().item())} ({100*active.float().mean().item():.2f}%) "
                                  f"|disp|(mean={disp.mean().item():.2e} med={disp.median().item():.2e} "
                                  f"max={disp.max().item():.2e}) "
                                  f"xyz_lr={float(xyz_lr):.2e} noise_lr={noise_lr:.1e} extent={gaussians.spatial_lr_scale:.3f}")
                            if bool(getattr(opt, "cov_noise", False)):
                                print(f"[NOISE it{iteration}] cov: s(mean={s.mean().item():.3e} "
                                      f"min={s.min().item():.3e} max={s.max().item():.3e}) "
                                      f"normal_std(mean={normal_std.mean().item():.3e}) pad={pad:.2f}")

                        gaussians._xyz.add_(noise)

                # Reset de opacidades periódico (default deshabilitado: interval=1e9).
                # GATE: solo se permite hasta UN intervalo de reset ANTES de cerrar la
                # densificación, para que tras el último reset la nube tenga al menos
                # `opacity_reset_interval` iters de recuperación (relocate/add_new +
                # gradiente) antes de la fase de consolidación. Ej.: densify_until=25000
                # y reset=3000 → último reset permitido @22000.
                # (ADVERTENCIA: opacity_reset abre la compuerta del ruido MCMC (1−o)^100
                #  para toda la nube en cada reset; ver desastre run10 en CLAUDE.md.)
                reset_cutoff = opt.densify_until_iter - opt.opacity_reset_interval
                if iteration <= reset_cutoff and (
                    iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)
                ):
                    print_memory("antes_reset")
                    gaussians.reset_opacity(iteration=iteration)
                    print_memory("después_reset")
            else:
                # Post-densify: sanear NaN/Inf cada densification_interval pasos para que
                # los últimos miles de iters no acumulen splats degenerados (zonas negras).
                if iteration % opt.densification_interval == 0:
                    gaussians.sanitize_parameters(iteration=iteration)

            # Optimizer step
            if iteration < opt.iterations:
                # NaN-safe: limpia gradientes con NaN/Inf antes de step. Evita que un
                # único batch con gradiente patológico contamine los parámetros para
                # siempre. Cubre TODAS las iters (no solo las de densificación).
                for p in gaussians.optimizer.param_groups:
                    for tensor in p['params']:
                        if tensor.grad is not None:
                            torch.nan_to_num_(tensor.grad, nan=0.0, posinf=0.0, neginf=0.0)

                # Debug del TÉRMINO 1 de la ecuación MCMC: μ ← μ − λ_lr·∇L (paso de Adam).
                # Capturamos _xyz antes/después del step para medir el desplazamiento REAL
                # por gradiente |Δμ_grad|, y el |∇_μL| crudo. Misma cadencia que DEBUG_NOISE
                # (en pasos de densificación) para comparar contra |disp| del ruido (término 2).
                report_grad = (
                    DEBUG_NOISE
                    and iteration % opt.densification_interval == 0
                    and (iteration // opt.densification_interval) % DEBUG_NOISE == 0
                )
                if report_grad:
                    xyz_before = gaussians._xyz.detach().clone()
                    g = gaussians._xyz.grad
                    gnorm = g.norm(dim=1) if g is not None else None

                gaussians.optimizer.step()

                if report_grad:
                    dgrad = (gaussians._xyz.detach() - xyz_before).norm(dim=1)
                    gn = ("none" if gnorm is None else
                          f"mean={gnorm.mean().item():.2e} max={gnorm.max().item():.2e}")
                    print(f"[GRAD it{iteration}] |grad_xyz|({gn}) "
                          f"|disp_grad|(mean={dgrad.mean().item():.2e} med={dgrad.median().item():.2e} "
                          f"max={dgrad.max().item():.2e}) "
                          f"xyz_lr={float(xyz_lr):.2e} extent={gaussians.spatial_lr_scale:.3f}")

                gaussians.optimizer.zero_grad(set_to_none = True)

            mem_probe("post_step", iteration, npts=gaussians.get_xyz.shape[0])

            # ✅ Clamp suave del parámetro b (no del beta)
            # TECHO SUBIDO (run73, 2026-07-21): max 2.0 -> 2.7081. beta = 4*e^_beta,
            # asi que 2.0 -> beta_techo 4*e^2 = 29.556 (viejo) y 2.7081 -> 4*e^2.7081 = 60 (nuevo).
            # DEBE ir de la mano con get_beta (gaussian_model.py:184), que tambien clampa
            # _beta a max=2.0 en el forward: si solo se sube este, get_beta sigue recortando
            # a 29.556 y el experimento es un NO-OP (fallo tipo run65). Min = -4.0 en AMBOS.
            with torch.no_grad():
                gaussians._beta.data.clamp_(min=-4.0, max=2.0)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        # [BETA] contador de splats con beta<0.1 (los "platos planos" que --freeze_low_beta
        # congela). Sirve para interpretar el A/B run30: si beta<0.1 es ~0%, el freeze es un
        # no-op por construcción y el experimento no concluye nada. Ver docs/beta_kernel_y_freeze_low_beta.html
        with torch.no_grad():
            _beta = scene.gaussians.get_beta.flatten()
            _n = _beta.numel()
            _low = int((_beta < 0.1).sum().item())
            print("\n[ITER {}] [BETA] total={} | beta<0.1: {} ({:.3f}%) | beta min/mean/max = {:.4f}/{:.4f}/{:.4f}".format(
                iteration, _n, _low, 100.0 * _low / max(_n, 1),
                _beta.min().item(), _beta.mean().item(), _beta.max().item()))
            # [BETA-TECHO] topados en el clamp SUPERIOR (run73). Leccion de
            # docs/techo_beta_clamp_superior.html: para saber si un techo MUERDE hay que
            # CONTAR topados, no mirar solo 'beta max' (que es un max sobre millones de
            # muestras y satura con que UN solo splat llegue). Sin esto, subir el techo de
            # 29.556 a 60 no se podria evaluar. beta>29.556 = cuantos superan el techo VIEJO.
            _braw = scene.gaussians._beta.detach().flatten()
            _ceil_raw = 2.7081                                  # max clamp -> beta_techo ~ 60
            _top = int((_braw >= _ceil_raw - 1e-3).sum().item())
            _over_old = int((_beta > 29.556).sum().item())
            print("[ITER {}] [BETA-TECHO] techo _beta={:.4f} (beta~60) | topados: {} ({:.4f}%) | beta>29.556 (techo viejo 4e2): {} ({:.4f}%)".format(
                iteration, _ceil_raw, _top, 100.0 * _top / max(_n, 1), _over_old, 100.0 * _over_old / max(_n, 1)))

        # [CLAMP] verifica que el techo de escala se aplique de verdad (%topados > 0)
        # y con qué factor. Detecta el fallo de run65: script "small clamp" sin la env
        # var exportada -> corrió con el default 0.1 sin avisar.
        scene.gaussians.clamp_report(iteration)

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)   
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
