export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN — DENSIFICACIÓN MCMC (relocate_gs + add_new_gs + ruido posicional)
# ═══════════════════════════════════════════════════════════════════════════
# Camino MCMC: --densify_mode mcmc. Las palancas vivas del MCMC son el cap, el
# gate de muerte (dead_sustain), las regs L1 (opacity_reg/scale_reg), el ruido
# (noise_lr/cov_noise) y el sesgo por error (mcmc_error_weight/jitter). El
# opacity_reset queda OFF (interval=1e9): reactivarlo abre la compuerta del ruido
# (1−o)^100 y rompe el MCMC (runs 10/11/12). Para el clásico → train_CLASSIC_server.sh.
#
# DEFAULTS = run16 (mejor honesto MCMC: 20.21 / 0.597 / 0.339). Editar el bloque de
# abajo para barrer parámetros. Tras el run: render_server.sh + metrics.py + fila al
# historial y a docs/comparativa_runs.html.
# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=23                   # nº de run → output/m360/${DATASET}_beta_run${RUN}

CAP_MAX=7500000          # techo de splats (flowers=7.5M base run16; cap saturado)
DEAD_SUSTAIN=5           # gate de muerte sostenida (run16=5=óptimo; >5 = más velo, monótono)
OPACITY_REG=0.01         # reg L1 opacidad (run16=0.01; run20=0.02 PEOR)
SCALE_REG=0.01           # reg L1 escala (run16=0.01; run21=0.05 afina foreground, no fondo)
NOISE_LR=3e3             # escala ruido posicional (config run9/run16)
MCMC_ERROR_WEIGHT=3.5    # sesgo de muestreo por error de reconstrucción
MCMC_JITTER_SCALE=1.5    # jitter de add_new_gs ∝ error·scale (siembra dentro de huecos)
OPACITY_CULL=0.01        # umbral de opacidad para dead_mask del relocate
FLOATER_CULL_DIST=0.2    # cull anti-floater por proximidad a cámaras de train (·extent)
COV_NOISE_NORMAL=1.0     # escala isotrópica de la componente normal del ruido híbrido
LAMBDA_DIST=10           # reg de distorsión 2DGS
LAMBDA_NORMAL=0.05       # reg de consistencia de normales
ITERATIONS=30000         # test pica ~30k y decae (early-stop confirmado)
DENSIFY_UNTIL=25000      # fin de densificación

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --densify_mode mcmc \
    --iterations $ITERATIONS \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_until_iter $DENSIFY_UNTIL \
    --lambda_normal $LAMBDA_NORMAL \
    --lambda_dist $LAMBDA_DIST \
    --opacity_reset_interval 1000000000 \
    --cap_max $CAP_MAX \
    --noise_lr $NOISE_LR \
    --scale_reg $SCALE_REG \
    --opacity_reg $OPACITY_REG \
    --opacity_cull $OPACITY_CULL \
    --floater_cull_dist $FLOATER_CULL_DIST \
    --mcmc_error_weight $MCMC_ERROR_WEIGHT \
    --mcmc_jitter_scale $MCMC_JITTER_SCALE \
    --cov_noise \
    --cov_noise_normal $COV_NOISE_NORMAL \
    --mcmc_dead_sustain $DEAD_SUSTAIN \
    2>&1 | tee $LOG
