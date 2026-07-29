export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000
export DEBUG_DENSIFY=1

# ═══════════════════════════════════════════════════════════════════════════
#  run4 — SOLO EL PRUNE POR MUNDO (fix B). El de pantalla, APAGADO.
# ═══════════════════════════════════════════════════════════════════════════
#
#  ⚠ NO LANZADO. El A/B local ya lo midió y la predicción es que PIERDE.
#  Dos runs de 10.000 iters (reloj del run real escalado x1/3, ~1,9M splats,
#  -r 4, único delta el prune de mundo), métricas honestas con metrics.py:
#        off   (= run2)  19.906 / 0.5266 / 0.4135
#        world (= esto)  18.475 / 0.5007 / 0.4340   <- peor en las TRES
#  podando solo 32.843 splats en 39 densificaciones (1,7% del modelo). El brazo
#  off queda cerca del run2 real de 30k (20.16/0.554/0.390), así que el smoke
#  reproduce el régimen bueno. Se conserva este script por si se quiere el dato
#  honesto a 30k de todos modos, pero SIZE_PRUNE_MODE ya tiene 'off' por
#  defecto y esa es la recomendación.
#

# Config IDÉNTICA a run2. run3 (= run2 + los dos prunes vivos) colapsó: PSNR
# honesto 8,8 contra 20,16. El post-mortem está en
# docs/prunes_de_tamano_explicado.html §8-§9 y se resume así:
#
#   El loss de run3 es idéntico al de run2 dígito a dígito hasta el
#   opacity_reset de la iteración 3000. En la densificación siguiente —la
#   primera con size_threshold activo— el prune por PANTALLA se lleva 406.924
#   splats de 1.141.501 (35%) y a partir de ahí 25-45k en cada una. El loss se
#   queda clavado en 0,4 (run2: 0,14), el modelo acaba con 12,9M splats en vez
#   de 6,8M y no se recupera nunca. El prune por MUNDO, en cambio, se lleva
#   13.598 en el primer golpe y entre 0 y 15 después: inofensivo.
#
#   Causa: el umbral de 20 px (train.py:228) viene del 3DGS original, donde
#   NUNCA llegó a ejecutarse por el mismo bug del reset. No lo ha calibrado
#   nadie. Medido sobre run2 (30 vistas, scripts/medir_radii_sin_clamp.py):
#   el 18,8% de los splats visibles supera 20 px de radio -> "grande en
#   pantalla" no es una anomalía, es el fondo de la escena.
#
# DELTAS RESPECTO A run2 (los tres, y solo estos):
#   1. SIZE_PRUNE_MODE=world -> el prune por mundo vive (fix B), el de pantalla
#      sigue apagado. Es el A/B que run3 no llegó a ser.
#   2. gaussian_renderer/__init__.py: QUITADO el `radii = clamp(radii, max=50)`.
#      No afecta al render (el binning ya se hizo con el radio real) y su único
#      consumidor era max_radii2D. Con el prune de pantalla OFF, no cambia una
#      sola métrica; lo que hace es que el log diga la verdad sobre los radios.
#   3. classic_prune_sustain=25, como run2 (run3 lo bajó a 15 -> segundo delta
#      que ensuciaba la comparación; aquí se devuelve al valor de run2).
#
# QUÉ MIRAR EN EL LOG (por orden):
#   1. [PRUNE-TAM] size_prune_mode = world (screen=OFF, world=ON) en el arranque.
#   2. [DENSIFY] -> world=N(thr=0.3371 s_max=...) con N>0 en las primeras
#      densificaciones tras iter 3000, y screen=OFF (no "0": OFF es apagado,
#      0 sería el criterio vivo sin encontrar nada = el bug de antes).
#   3. [RADIO2D] -> ESTE es el objetivo secundario del run. Sin el radius_clip,
#      p50/p90/p99/max y los conteos >20/>50/>100 px son reales. Es la tabla que
#      hace falta para calibrar max_screen_size en un run futuro. Referencia
#      medida sobre run2 acabado: p50=10 p90=33 p99=100 p99,9=286 max=1600,
#      con >20px = 18,8% y >100px = 0,99% de los visibles.
#   4. Nº de splats final vs los 6,88M de run2: el prune por mundo debería
#      bajarlo un poco (run3 mató 13.598 de golpe en el primer disparo).
#   5. [CLAMP] recorte_max = 0.000000 y [A] VIOLACIONES neg/sum = 0/0, igual que
#      en run2 (esos fixes no se han tocado).
#
# COMPARACIÓN HONESTA (metrics.py, honesto-vs-honesto):
#   run2 (prunes de tamaño muertos)  20.1602 / 0.5537 / 0.3900   <- el A/B directo
#   run3 (los dos vivos, umbral 20)   ~8.8    (colapso)
#   run67 (clásico, kernel tent)     20.6684 / 0.5811 / 0.3675
#   baseline 2DGS oficial            20.8900 / 0.5560 / 0.4020
# ───────────────────────────────────────────────────────────────────────────
DATASET=flowers
RUN=4

export SCALE_CLAMP_FACTOR=0.1 # = run2/run3
export WS_PRUNE_FRACTION=0.7  # umbral del prune por mundo = 0.7 * techo = 0.3371
export SIZE_PRUNE_MODE=world  # <<< EL DELTA: screen OFF, world ON

PRUNE_SUSTAIN=25              # = run2 (run3 usó 15)
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

# Tras el run: render_server.sh (RUN=4) + metrics.py -m $MODEL + fila al historial.
# Y guardar la tabla de [RADIO2D]: es la calibración de max_screen_size para después.
#   grep "\[RADIO2D" logs/flowers4.log | tail -20
