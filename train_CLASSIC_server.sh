export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)
export DEBUG_DENSIFY=1   # imprime [DENSIFY]/[RESET] (default ON; =0 silencia)
# (DEBUG_NOISE NO aplica: el ruido posicional MCMC está OFF en el camino clásico)

# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN — DENSIFICACIÓN CLÁSICA 2DGS (clone/split + prune + opacity_reset)
# ═══════════════════════════════════════════════════════════════════════════
# Camino clásico: --densify_mode classic. densify_and_prune dirigido por
# ‖∂L/∂μ₂D‖ (clone bajo under-recon + split bajo over-recon) + opacity_reset
# periódico. El ruido posicional MCMC, el cull de floaters y mcmc_* NO se aplican
# (son del camino MCMC, el clásico los ignora). cap_max tampoco aplica: el clásico
# crece libre → VIGILAR nº de splats por OOM. Para el MCMC → train_MCMC_server.sh.
#
# OJO opacity_reset: en clásico es SEGURO (3000 = ciclo nativo reset→recupera/prune;
# el colapso de runs 10/11/12 era la compuerta (1−o)^100 del MCMC, OFF aquí). En MCMC
# rompe. Palanca propia del clásico: classic_prune_sustain (prune sostenido N pasos).
#
# DEFAULTS = run15 (mejor clásico honesto: 20.13 / 0.548 / 0.387; prune inmediato +
# FIX load_ply beta). Modos de error opuestos al MCMC: el clásico hace HUECOS NEGROS
# (sub-cobertura del fondo rasante), el MCMC hace VELO translúcido. Tras el run:
# render_server.sh + metrics.py + fila al historial y a docs/comparativa_runs.html.
# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers               # carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=67                        # nº de run → output/m360/${DATASET}_beta_run${RUN}
# ═══ run67 = ANCLA run26 (clásico sano) + UN SOLO DELTA: el fix del trinquete de β ═══
# Deltas vs run66 (los 2 primeros REVIERTEN la regresión medida, no son experimento):
#   · OPACITY_REG 0.02→0 y SCALE_REG 0.06→0  = reguladores de run25/26. A/B local midió
#     que scale_reg 0.06 cuesta 1.68 dB y opacity_reg 0.02 otros 0.14 dB en clásico.
#   · PRUNE_SUSTAIN 5→25 = igual que run26, para que el ancla sea exacta.
# DELTA NUEVO (el experimento): gaussian_model.py:574 ya NO resta math.log(N) al crear
#   los hijos del split → β se hereda tal cual (docs/beta_trinquete_split_clasico.html).
# ANCLA de comparación = run26: 19.99 / 0.537 / 0.394 honesto, in-train train@30k 22.22.
#
# CULL_SUBPIXEL=1 SE MANTIENE (decisión del usuario 2026-07-20: seguir ese experimento).
#   Consecuencia: run67 tiene DOS deltas vs run26 (fix de beta + cull), porque run25/26
#   son PRE-cull (build del cull = 2026-07-05). PERO el fix de beta YA ESTA AISLADO por
#   el A/B local (flowers, 2500 it, regs=0, prune_sustain=25) con CULL=1 en AMBOS brazos:
#       con trinquete: train 18.84 | beta<0.1 = 57.4% | beta min 0.0733 (suelo)
#       con el fix   : train 19.31 | beta<0.1 =  0.0% | beta min 0.5110
#   -> +0.47 dB y colapso de beta ELIMINADO, sin NaN. El cull es comun a los dos brazos,
#      asi que no contamina esa medida. Lo que run67 anade es la medida HONESTA a 30k.
#   OJO al interpretar: comparar el ABSOLUTO de run67 contra run26 mezcla los 2 deltas.
#   Si hace falta el single-delta a 30k, el run que falta es "clasico + CULL + regs=0 +
#   trinquete" (= run67 sin el fix), no desactivar el cull.

# ⚠ run65 se tituló "small clamp" pero NUNCA exportó esta env var → corrió con el
# default 0.1. Dejarla explícita aquí + el print [CLAMP] evita repetir el fallo.
export SCALE_CLAMP_FACTOR=0.1  # 0.1 = default 3DGS/2DGS; run64 probó 0.05

PRUNE_SUSTAIN=25              # CONFIG GANADORA = run67 (MEJOR CLÁSICO: 20.6684/0.5811/0.3675 honesto).
                              # Barrido LIMPIO (β [-4,2], trinquete muerto) CERRADO: ps 5=20.5439(run74)
                              # · 10=20.7436(run72) · 20=20.6318(run75) · 25=20.6684(run67) → plano
                              # dentro de ~0.20 dB, SSIM/LPIPS clavados → prune_sustain NO es palanca.
OPACITY_RESET_INTERVAL=3000   # ciclo reset→recupera/prune (SEGURO en clásico; del 2DGS original)
DENSIFY_FROM=500              # inicio de densificación
DENSIFY_UNTIL=15000           # fin de densificación (ventana del 2DGS original)
DENSIFICATION_INTERVAL=100    # cada cuántas iters se densifica/poda
DENSIFY_GRAD_THRESHOLD=0.0002 # umbral de ‖∂L/∂μ₂D‖ para clone/split
PERCENT_DENSE=0.01            # umbral de tamaño clone vs split
OPACITY_CULL=0.005             # min_opacity del prune (0.005=2DGS original; 0.01=poda más agresiva, run65)
LAMBDA_DIST=0                 # reg distorsión (0 = receta 2DGS original; el 10 era nuestro)
LAMBDA_NORMAL=0.05            # reg de consistencia de normales
OPACITY_REG=0                 # OFF = run25/26. En clásico la L1 cuesta PSNR (A/B local: 0.02 -> -0.14 dB)
SCALE_REG=0                   # OFF = run25/26. A/B local: 0.06 cuesta -1.68 dB en clásico
ITERATIONS=30000

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --densify_mode classic \
    --iterations $ITERATIONS \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_from_iter $DENSIFY_FROM \
    --densify_until_iter $DENSIFY_UNTIL \
    --densification_interval $DENSIFICATION_INTERVAL \
    --densify_grad_threshold $DENSIFY_GRAD_THRESHOLD \
    --percent_dense $PERCENT_DENSE \
    --opacity_reset_interval $OPACITY_RESET_INTERVAL \
    --opacity_cull $OPACITY_CULL \
    --lambda_normal $LAMBDA_NORMAL \
    --lambda_dist $LAMBDA_DIST \
    --opacity_reg $OPACITY_REG \
    --scale_reg $SCALE_REG \
    --classic_prune_sustain $PRUNE_SUSTAIN \
    2>&1 | tee $LOG
