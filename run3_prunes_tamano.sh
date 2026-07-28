export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000
export DEBUG_DENSIFY=1

# ═══════════════════════════════════════════════════════════════════════════
#  run3 — RE-RUN DE run2 CON LOS DOS PRUNES DE TAMAÑO ARREGLADOS
# ═══════════════════════════════════════════════════════════════════════════
# Config IDÉNTICA a run2 (kernel Gabor con a_n>=0 y sum(a_n)<=1/2 + los dos
# fixes del clamp de escala). Único delta: los fixes A y B de
# docs/prunes_de_tamano_explicado.html. NO se toca el kernel -> A/B limpio
# contra run2 (20.1602 / 0.5537 / 0.3900).
#
#   FIX A — scene/gaussian_model.py: densification_postfix ya NO resetea
#     max_radii2D a ceros; ahora CONCATENA ceros solo para los splats nuevos
#     (mismo patrón que low_opacity_counter). Antes, densify_and_prune llamaba
#     a clone/split ANTES de leer big_points_vs, así que el prune por radio en
#     pantalla leía siempre [0,0,...] y "0 > 20" es False -> código muerto
#     (screen=0 en las 144 densificaciones de run2). prune_points ya filtraba
#     max_radii2D, así que el tensor sigue alineado aunque densify_and_split
#     reordene los splats al podar los padres. La ventana de acumulación se
#     reinicia al final de densify_and_prune, ya usado el dato.
#
#   FIX B — scene/gaussian_model.py: el umbral del prune por mundo era
#     0.1*extent, EXACTAMENTE el techo del clamp de get_scaling
#     (scale_clamp_factor*spatial_lr_scale, con extent = spatial_lr_scale).
#     "clamp(x, max=C) > C" es False siempre -> el segundo prune también estaba
#     muerto (world=0 en las 144). Ahora el umbral es
#     min(0.1*extent, WS_PRUNE_FRACTION * techo) = 0.07*extent por defecto, por
#     DEBAJO del techo. WS_PRUNE_FRACTION es env var para A/B del agresividad.
#
# QUÉ MIRAR EN EL LOG (por orden):
#   1. [DENSIFY] -> screen=N(thr=20 r2d_max=...) y world=N(thr=... s_max=...).
#      Si vuelven a salir screen=0 y world=0 en TODAS las densificaciones, el
#      prune se ha vuelto a morir. Ojo: screen solo puede disparar DESPUÉS del
#      primer opacity_reset (size_threshold=None hasta iter > 3000).
#   2. s_max debe quedarse pegado al techo (0.481604 con SCALE_CLAMP_FACTOR=0.1)
#      pero ahora esos topados MUEREN: world>0 en cuanto haya gigantes.
#   3. Nº de splats final vs run2 (6.883.777 en run1): debería BAJAR — se está
#      podando lo que antes sobrevivía. Si baja demasiado y aparecen huecos
#      negros, subir WS_PRUNE_FRACTION (0.8/0.9) es el dial.
#   4. [CLAMP] recorte_max = 0.000000 y [A] VIOLACIONES neg/sum = 0/0, igual
#      que en run2 (esos fixes no se han tocado).
#
# COMPARACIÓN HONESTA (metrics.py, honesto-vs-honesto):
#   run2 (mismo código, prunes muertos)  20.1602 / 0.5537 / 0.3900
#   run67 (clásico, kernel tent)         20.6684 / 0.5811 / 0.3675
#   baseline 2DGS oficial                20.8900 / 0.5560 / 0.4020
# ───────────────────────────────────────────────────────────────────────────
DATASET=flowers
RUN=3                         # run2 se conserva intacto como referencia

export SCALE_CLAMP_FACTOR=0.1 # = run2/run1/run67
export WS_PRUNE_FRACTION=0.7  # FIX B: umbral del prune por mundo = 0.7 * techo

PRUNE_SUSTAIN=25              # todo lo de abajo = run2, sin tocar
OPACITY_RESET_INTERVAL=3000
DENSIFY_FROM=500
DENSIFY_UNTIL=15000
DENSIFICATION_INTERVAL=100
DENSIFY_GRAD_THRESHOLD=0.0002
PERCENT_DENSE=0.01
OPACITY_CULL=0.005
LAMBDA_DIST=0
LAMBDA_NORMAL=0.05
OPACITY_REG=0
SCALE_REG=0
ITERATIONS=30000

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --densify_mode classic \
    --iterations $ITERATIONS \
    --test_iterations 2500 7000 15000 20000 25000 30000 \
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

# Tras el run: render_server.sh (RUN=3) + metrics.py -m $MODEL + fila al historial.
