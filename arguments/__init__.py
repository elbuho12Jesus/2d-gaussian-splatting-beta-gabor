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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        # --- Spherical Betas (Beta Splatting oficial) ---
        # Nº de lóbulos SB por splat para color view-dependent:
        #   C = SH(dir) + Σᵢ cᵢ · max(dot(μᵢ, dir), 0)^(4·exp(bᵢ))
        # Cada lóbulo = 6 params (r,g,b,θ,φ,b). El oficial usa sb_number=2 con
        # sh_degree=0 (solo DC): 3+12=15 params de color vs 48 de SH3, y los lóbulos
        # capturan el view-dependence que SH3 no puede (cielo-entre-hojas, césped
        # rasante). 0 = desactivado (compat con runs previos, color 100% SH).
        self.sb_number = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        # A/B FIX #2 (run9): si True, congela geometría/opacidad de splats con
        # beta<0.1 en el backward (comportamiento PRE-run9). Default False = run9.
        self.freeze_low_beta = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        # lr de los lóbulos Spherical Beta (default oficial beta-splatting)
        self.sb_params_lr = 0.0025
        self.opacity_lr = 0.05
        self.beta_lr = 0.001
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.cap_max = 100000
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.005
        self.scale_reg = 0.01
        self.opacity_reg = 0.01

        self.densification_interval = 100
        self.opacity_reset_interval = 1_000_000_000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        # --- Ruido posicional MCMC (alineado con Beta Splatting oficial) ---
        # Exponente para el multiplicador de ruido: (1 - opacity) ** exp
        self.noise_opacity_exponent = 100.0
        # Factor de escala del ruido. Default oficial = 5e4 (se multiplica por xyz_lr ~1e-6).
        # Histórico: 5e5 producía desplazamientos ~0.5 unidades world en splats
        # casi-muertos → NaN en _opacity (Run 3). Bajado a 1e5 estabilizó, y a 5e4
        # en Run 8 para alinear con el oficial y reducir desperdicio de splats.
        self.noise_lr = 5e4
        # --- Muestreo MCMC sesgado por error de reconstrucción ---
        # Blend constante: probs ∝ opacity · (1 + mcmc_error_weight · error_norm),
        # donde error_norm = gradiente de viewspace acumulado por splat (proxy 3DGS de
        # zona sub-reconstruida), normalizado por su media. 0 = muestreo por opacidad
        # pura (comportamiento previo). Sube presupuesto a bordes de zonas mal cubiertas
        # (huecos negros de césped/cielo) sin perder el detalle de foreground.
        self.mcmc_error_weight = 2.0
        # Jitter posicional en add_new_gs proporcional a error × scale del src.
        # Sirve para sembrar splats DENTRO de huecos vacíos: sin jitter, add_new_gs
        # clona en la misma posición que el src (que vive en el BORDE del hueco
        # donde el error sube), así que el clon también nace en el borde y el
        # interior nunca se rellena. Con jitter_scale > 0, los srcs de alto error
        # producen clones desplazados ~ direction · scale_src · (jitter_scale · err).
        # 0 = sin jitter (comportamiento previo). Solo se activa si error_weight > 0.
        self.mcmc_jitter_scale = 0.0
        self.beta_densify_threshold = 0.0   # 0 = desactivado, >0 = umbral activado
        self.beta_densify_mode = "split_wide"  # o "split_narrow"
        # --- Ruido posicional híbrido por covarianza (surfel 2D) ---
        # False = ruido isotrópico (default). True = anisotropía moldeada por las 2
        # escalas EN EL PLANO del surfel + isotrópico en la normal (ver bloque de ruido
        # en train.py y docs/ruido_isotropico_vs_covarianza.html, secciones 5/8).
        self.cov_noise = False
        # Escala isotrópica de la componente normal del ruido híbrido. 1.0 = la normal
        # explora a la tasa media del plano; 0.0 = ruido CONFINADO al plano (rango 2).
        self.cov_noise_normal = 1.0
        # --- Cull anti-floater por proximidad a cámaras de train ---
        # Marca como muertos (→ relocate MCMC) los splats a menos de
        # floater_cull_dist · extent de CUALQUIER centro de cámara de train.
        # Medido en flowers4 (7.5M): el 99% de los splats está a >0.25·extent de
        # toda cámara; la cola <0.2·extent (~49k, op_med 0.25) son floaters que
        # emborronan el césped en vistas nuevas. 0.0 = desactivado (default).
        self.floater_cull_dist = 0.0
        # --- Gate de muerte sostenida para el relocate MCMC ---
        # 0 (default) = comportamiento original: dead_mask = (opacidad <= opacity_cull)
        # INSTANTÁNEO → relocate recicla cualquier splat que baje del cull aunque sea un
        # instante. >0 = exige que el splat lleve `mcmc_dead_sustain` checks de densify
        # (cada densification_interval iters) CONSECUTIVOS bajo el cull antes de reubicarlo
        # (low_opacity_counter > N). Da a los splats del fondo, que fluctúan, tiempo de
        # recuperar opacidad antes de ser reciclados. Viable en MCMC porque NO hay
        # opacity_reset (interval=1e9) que borre el contador; requiere el fix de
        # densification_postfix que preserva el contador. Ver docs/low_opacity_counter_y_reset.html.
        self.mcmc_dead_sustain = 0
        # --- Densificación clásica 2DGS (clone/split) en vez de MCMC ---
        # False (default) = densificación MCMC (relocate_gs + add_new_gs + ruido).
        # True = densificación clásica del 2DGS/3DGS original: densify_and_prune
        # (clone para under-reconstrucción + split para over-reconstrucción, dirigido
        # por gradiente de viewspace) + opacity_reset periódico. Con classic_densify=True
        # el ruido posicional MCMC y el cull de floaters NO se aplican (son del MCMC).
        # Prueba para aislar si el garabato del fondo es del MCMC. Ver
        # docs/clone_split_logica_huecos.html y docs/mcmc_beta_vs_clonesplit.html.
        # NOTA: alias retrocompatible de --densify_mode. Se mantiene para los scripts
        # existentes (train_server.sh/train.sh) que ya lo pasan. Si se pasa
        # --classic_densify, FUERZA el modo clásico aunque densify_mode diga "mcmc".
        self.classic_densify = False
        # --- Selector EXPLÍCITO de algoritmo de densificación (interfaz preferida) ---
        # "mcmc" (default) = relocate_gs + add_new_gs + ruido posicional MCMC.
        # "classic"        = densify_and_prune (clone/split 2DGS) + opacity_reset.
        # Es la cara legible de classic_densify (evita el bool de doble negación).
        # Resolución en train.py: classic ⇔ (densify_mode=="classic") OR classic_densify.
        # Las dos rutas están aisladas (if/elif/else en train.py): cambiar de modo NO
        # afecta a la otra. Ver docs/mcmc_beta_vs_clonesplit.html.
        self.densify_mode = "mcmc"
        # --- Prune SOSTENIDO por opacidad en la densificación CLÁSICA ---
        # 0 (default) = prune inmediato del 2DGS original: opacidad < cull se poda en el
        # acto (config run15). >0 = el splat debe llevar N pasos de densify CONSECUTIVOS
        # bajo el cull (low_opacity_counter > N) antes de podarse → le da N·densification_interval
        # iters para "asentarse"/recuperar opacidad por gradiente antes de reciclarlo.
        # OJO: requiere opacity_reset OFF (o intervalo >> N·densification_interval); si el
        # reset cae cada M densifies con M<=N, sube todo a 0.01>cull → borra el counter →
        # criterio jamás se cumple = CÓDIGO MUERTO (caso run14, N=50 con reset cada 30).
        # Solo afecta al camino clásico (el MCMC usa --mcmc_dead_sustain).
        self.classic_prune_sustain = 0
        # ---------------------------------------------------
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
