export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000
export DEBUG_DENSIFY=1

# ═══════════════════════════════════════════════════════════════════════════
#  run2 — RE-RUN DE run1 CON LOS DOS FIXES DEL CLAMP DE ESCALA
# ═══════════════════════════════════════════════════════════════════════════
# Config IDÉNTICA a run1 (= ancla run67 clásico + kernel Gabor con a_n>=0 y
# sum(a_n)<=1/2). Único delta: los dos fixes del clamp de escala del 2026-07-26.
# Ver docs/clamp_render_vs_train.html.
#
#   FIX A — scene/__init__.py: load_ply ahora fija spatial_lr_scale = cameras_extent.
#     Antes se quedaba en 0 y el guard de get_scaling DESACTIVABA el techo al
#     renderizar -> render.py/metrics.py/visor pintaban la escala CRUDA del ply.
#     MEDIDO en run1 (mismo ply, dos renders locales, A/B limpio):
#         sin fix  19.9887 / 0.5531 / 0.3920   (= lo que dio el servidor)
#         con fix  20.0205 / 0.5530 / 0.3918   -> PSNR +0.042 dB
#     No afecta al entrenamiento, solo a la medida.
#
#   FIX B — train.py: _scaling.data.clamp_(max=log(techo)) tras cada paso, igual
#     que ya se hacía con _beta. Mata el trinquete: torch.clamp no propaga
#     gradiente por encima del máximo, así que un splat cuyo CRUDO estaba arriba
#     no volvía a bajar nunca. Dos vías de entrada medidas en run1:
#       1) nacer grande: el 5,89% de los 38.347 puntos SfM de flowers nace por
#          encima del techo (el mayor a 42x) -> congelados desde la iteración 0.
#       2) cruzar entrenando: Adam pasa de largo la barrera (simulado 1,073x;
#          p90 real 1,068x).
#     ESTE es el delta que exige re-entrenar: cambia lo que se guarda en el ply y
#     devuelve gradiente a los topados (pueden encogerse si la pérdida lo pide).
#
# QUÉ MIRAR EN EL LOG (por orden):
#   1. [CLAMP] -> recorte_max debe ser 0.000000 en TODAS las iters y s_raw max
#      == techo exacto (0.481604). Si recorte_max > 0, el fix B no está actuando.
#   2. [CLAMP] -> % topados: ahora es "cuántos están APOYADOS en el techo", no
#      "cuántos se han escapado". Puede subir sin ser malo; lo que ya no puede
#      pasar es que s_raw se despegue del techo.
#   3. [A] -> VIOLACIONES neg/sum = 0/0 (restricciones del kernel Gabor intactas).
#   4. [DENSIFY] -> grad_pos con mean y med del mismo orden (la firma del
#      reventón del run1 viejo era mean 4 órdenes por encima de med).
#   5. Nº de splats al final: run1 acabó en 6.883.777. Si B libera escala hacia
#      abajo, puede cambiar (splats más chicos -> más clones para cubrir).
#
# COMPARACIÓN HONESTA (todo con metrics.py y CON el fix A):
#   run1 (mismo código, sin los fixes)  20.0205 / 0.5530 / 0.3918  <- re-medido
#   run67 (clásico, kernel tent)        20.6684 / 0.5811 / 0.3675  <- OJO: medido
#       SIN el fix A. Para un A/B limpio del kernel hay que re-renderizar run67
#       con el fix (render + metrics, no hace falta re-entrenar).
#   baseline 2DGS oficial               20.8900 / 0.5560 / 0.4020  <- ídem
# ───────────────────────────────────────────────────────────────────────────
DATASET=flowers
RUN=2                         # run1 se conserva intacto como referencia

export SCALE_CLAMP_FACTOR=0.1 # = run1/run67. Con el fix B este techo ahora
                              # también acota el PARÁMETRO, no solo el forward.

PRUNE_SUSTAIN=25              # todo lo de abajo = run1 = ancla run67, sin tocar
OPACITY_RESET_INTERVAL=3000
DENSIFY_FROM=500
DENSIFY_UNTIL=15000
DENSIFICATION_INTERVAL=100
DENSIFY_GRAD_THRESHOLD=0.0002
PERCENT_DENSE=0.01
OPACITY_CULL=0.005
LAMBDA_DIST=0
LAMBDA_NORMAL=0.05
OPACITY_REG=0                 # OJO scale_reg/opacity_reg: train.py:156 los aplica
SCALE_REG=0                   # SOLO dentro de la ventana de densificación
                              # (500 < it < 15000) y sobre get_scaling, que ya
                              # viene CLAMPEADA -> gradiente cero para los
                              # topados. scale_reg nunca pudo arreglar esto.
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

# Tras el run: render_server.sh (RUN=2) + metrics.py -m $MODEL + fila al
# historial. El fix A ya está dentro, así que el render sale clampeado solo.
