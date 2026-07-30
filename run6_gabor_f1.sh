#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000
export DEBUG_DENSIFY=1

# ═══════════════════════════════════════════════════════════════════════════
#  run6 / run7 — SWEEP DE GABOR_F1 sobre el kernel ÁTOMO
#
#  USO:   ./run6_gabor_f1.sh 2      -> run6, f1 = 2× el autocalibrado (93.2494)
#         ./run6_gabor_f1.sh 4      -> run7, f1 = 4× el autocalibrado (186.4988)
# ═══════════════════════════════════════════════════════════════════════════
#
#  = run5_gabor_atomo.sh EXACTO + UN SOLO DELTA: GABOR_F1.
#
#  POR QUÉ (medido en run5, honesto 20.6898/0.5759/0.3724):
#  el átomo recuperó los −0,51 dB que perdía el kernel legacy y GANA EN LAS TRES a
#  run2_gabor (+0,53 dB), pero EMPATA con run67 (el tent) en vez de superarlo. El log
#  dice por qué: a 30 k el 93,40 % de los splats tiene la onda APAGADA (f·W < 0,5) y
#  solo el 2,91 % encendida. Para casi toda la nube el kernel *es* el tent, así que
#  empatar con el tent es exactamente lo esperado. No es que aprender la forma no
#  sirva: es que la forma casi no se está usando.
#
#  La causa del apagado está medida: f1 se autocalibra con las escalas INICIALES
#  (s_p90 = 0,36666 sobre los 38.347 puntos SfM) y al densificar los surfels encogen,
#  así que f·W CAE durante el run en vez de mantenerse:
#      f·W p50:  0,138 (@2500)  ->  0,102 (@7000)  ->  0,082 (@30000)
#      apagados: 88,62 %        ->  94,29 %        ->  93,40 %
#  Subir f1 mueve el histograma entero a la derecha y es la palanca más barata que
#  queda: no toca CUDA, no recompila, no cambia el número de parámetros.
#
#  QUÉ MIRAR EN EL LOG
#   1. [GABOR] al arrancar: f1 = el valor de abajo y "<- env GABOR_F1" (NO
#      "autocalibrado"). Con la env var puesta la autocalibración NO corre.
#   2. [GABOR-W] en cada test_iteration: el objetivo es bajar el % de APAGADOS
#      MANTENIENDO el eval. Se mira el HISTOGRAMA (p10/p50/p90), nunca el máximo.
#      · Si apagados baja y el eval sube -> la hipótesis se confirma, seguir subiendo.
#      · Si apagados baja y el eval BAJA -> f1 se pasó: la onda oscila más rápido que
#        el detalle de la escena y mete ruido de alta frecuencia. 2× sería el óptimo.
#   3. [A] LA SEGUNDA VÍA DE ESCAPE, que en run5 sí se activó: el optimizador también
#      apaga la onda por AMPLITUD. sum(a_n)<0,05 (kernel casi plano) subió 9,17 % ->
#      21,44 % durante el run, con a1 BAJANDO (0,1598->0,1243) y a3 subiendo. Si al
#      subir f1 la masa se va toda a Σa≈0, la palanca siguiente NO es más frecuencia:
#      es el init de a o un regularizador que penalice Σa->0.
#   4. beta es aprendible y SUBIRLO también apaga la onda (W = 2·s·σ_env(β)): vigilar
#      que [BETA] no se dispare respecto a run5 (mean 2,66-2,74 en los clásicos sanos).
#   5. [CLAMP] recorte_max = 0.000000 y VIOLACIONES neg/sum 0/0, como run5.
#
#  BASELINE HONESTO A BATIR (metrics.py, honesto-vs-honesto):
#    run2_gabor (legacy)              20.1602 / 0.5537 / 0.3900
#    run5_gabor (átomo, f1 = 1×)      20.6898 / 0.5759 / 0.3724   <- EL A/B DIRECTO
#    run67 (clásico, kernel tent)     20.6684 / 0.5811 / 0.3675   <- el punto neutro
#    baseline 2DGS oficial            20.8900 / 0.5560 / 0.4020
#    run36 (MCMC, techo global)       21.1600 / 0.5926 / 0.3555
# ───────────────────────────────────────────────────────────────────────────

MULT=${1:-2}
case "$MULT" in
  2) RUN=6; export GABOR_F1=93.2494  ;;   # 2 × 46.6247
  4) RUN=7; export GABOR_F1=186.4988 ;;   # 4 × 46.6247
  *) echo "Uso: $0 [2|4]   (multiplicador de f1 sobre el autocalibrado 46.6247)"; exit 1 ;;
esac
echo "════ run${RUN}: GABOR_F1=${GABOR_F1} (${MULT}× el autocalibrado de run5) ════"

DATASET=flowers

# ── EL DELTA: solo f1. Todo lo demás = run5 ──────────────────────────────
export GABOR_KERNEL=dir     # legacy | radial | dir
export GABOR_FREQ=world     # norm | world
export GABOR_GAMMA=0.3      # pedestal, = AdaGaR. sum(a_n) <= 1/(2-gamma) = 0.5882
export GABOR_KAPPA=1.0      # 1.0 = sin acotar el contraste

# ── Todo lo demás = run5_gabor / run2_gabor ──────────────────────────────
export SCALE_CLAMP_FACTOR=0.1
export SIZE_PRUNE_MODE=off    # medido: los dos prunes de tamaño restan (run3/run4)

PRUNE_SUSTAIN=25
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

# ── Después ────────────────────────────────────────────────────────────────
# El render ya NO necesita que recuerdes nada: render_server.sh lee las env vars
# de logs/${DATASET}${RUN}.log y verifica que el kernel del render coincide con el
# del train. Solo hay que poner RUN=${RUN} en su bloque de arriba:
#     ./render_server.sh && python metrics.py -m $MODEL
# y añadir la fila al historial.
#
# NOTA: con GABOR_F1 por env var NO se imprime la línea "f1 autocalibrado = ...";
# el valor sale en la línea [GABOR] de arranque como "f1=${GABOR_F1} <- env GABOR_F1".
# render_server.sh contempla los dos casos.
