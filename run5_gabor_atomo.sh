export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000
export DEBUG_DENSIFY=1

# ═══════════════════════════════════════════════════════════════════════════
#  run5 — KERNEL GABOR COMO ÁTOMO: envolvente × onda
# ═══════════════════════════════════════════════════════════════════════════
#
#  Config = run2_gabor EXACTO (la del kernel Gabor, clásico limpio) + el
#  rediseño del kernel. Análisis completo, ecuaciones y gráficos:
#      docs/rediseno_kernel_gabor_adagar.html
#
#  EL PROBLEMA QUE ARREGLA (medido sobre los ply de run2/run4 con
#  scripts/medir_kernel_gabor.py):
#    - El kernel legacy (f/f0)^beta NO tiene envolvente: el decaimiento y la
#      oscilación salen del MISMO sum(a_n)cos, así que los coeficientes hacen
#      dos trabajos con un solo presupuesto. Con a=0 el kernel es una CAJA
#      (disco plano), y ese punto está en la frontera a_n>=0.
#    - Reproducir la envolvente (la tent) consume el 93,3% de sum(a)<=1/2.
#    - Solo el 7,9% del conjunto factible da un perfil decreciente (70,8% con
#      pedestal). Resultado: 85,4% de splats no monótonos, contraste
#      valle/pico mediano 0,040 (el valle cae al 4% del pico) = las dianas.
#    - run4 aprendió a3 > a2 (espectro CRECIENTE, la tent decae 1/n^2): el
#      modelo intentó ser un Gabor por su cuenta, con geometría radial
#      (anillos) y sin interruptor de escala.
#
#  EL KERNEL NUEVO:
#      kernel = (1-r)^beta * S(phi*t)
#      S      = b + sum_n a_n cos((2n-1) phi t)
#      b      = gamma + (1-gamma)(1 - sum a_n)      <- pedestal (cambio 1)
#      t      = u  (coordenada del eje mayor)       <- direccional (cambio 3)
#      phi    = f1 * s_u / extent                   <- unidades de MUNDO (cambio 2)
#      sum(a_n) <= kappa/(2-gamma)                  <- dial de contraste (cambio 4)
#
#  a = 0  =>  b = 1  =>  S == 1  =>  kernel = (1-r)^beta = el tent de run67
#  EXACTO. El init arranca AHÍ, así que el baseline es el punto neutro del
#  espacio y aprender la forma solo puede sumar.
#
#  QUÉ MIRAR EN EL LOG
#   1. [GABOR] al arrancar: kernel=dir, freq=world, y el f1 autocalibrado.
#      ANOTAR ESE f1: hay que exportarlo en el render o el kernel no será el
#      mismo (misma clase de bug que los haces de luz de run64).
#   2. [GABOR-W] cada test_iteration: percentiles de f*W y % de splats
#      APAGADOS (f*W<0.5). Si la masa se va a f*W<0.5, la onda está apagada y
#      las métricas no miden lo que se cree. beta es aprendible y subirlo
#      apaga la onda: es la vía de escape barata del optimizador.
#   3. CONTROL DE ARRANQUE: el eval in-train de la iteración 2500 debe quedar
#      cerca del de run67 (tent), NO por debajo del de run2_gabor. Si arranca
#      peor que el tent, algo va mal en el init o en el pedestal.
#   4. [A] sum(a_n) <= 0.5882 (= 1/(2-gamma), no 0.5) y VIOLACIONES 0/0.
#   5. [CLAMP] recorte_max = 0.000000, igual que run2.
#
#  COMPARACIÓN HONESTA (metrics.py, honesto-vs-honesto):
#    run1_gabor                       20.0205 / 0.5530 / 0.3918
#    run2_gabor (legacy + fixes)      20.1602 / 0.5537 / 0.3900   <- el A/B directo
#    run4_gabor (+ prune de mundo)    18.1817 / 0.5178 / 0.4171
#    run67 (clásico, kernel tent)     20.6684 / 0.5811 / 0.3675   <- el baseline del kernel
#    baseline 2DGS oficial            20.8900 / 0.5560 / 0.4020
#    run36 (MCMC, techo global)       21.1600 / 0.5926 / 0.3555
# ───────────────────────────────────────────────────────────────────────────
DATASET=flowers
RUN=5

# ── EL DELTA: el kernel ──────────────────────────────────────────────────
export GABOR_KERNEL=dir     # legacy | radial | dir   (dir = cambios 1+3)
export GABOR_FREQ=world     # norm | world           (world = cambio 2)
export GABOR_GAMMA=0.3      # pedestal, = AdaGaR. sum(a_n) <= 1/(2-gamma) = 0.5882
export GABOR_KAPPA=1.0      # cambio 4: 1.0 = sin acotar el contraste
# GABOR_F1 sin exportar = autocalibrado a f*W=1 en el p90 de las escalas
# iniciales. ANOTAR el valor que imprima [GABOR] y exportarlo en el render.

# ── Todo lo demás = run2_gabor ───────────────────────────────────────────
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

# ⚠ EL RASTERIZER HAY QUE RECOMPILARLO: el tensor `a` pasó de (N,3) a (N,5)
#   y hay un flag nuevo (gabor_mode) en la firma.
#     rm -rf submodules/diff-surfel-rasterization/build
#     pip install submodules/diff-surfel-rasterization
#
# Tras el run, para render + metrics, EXPORTAR LAS MISMAS ENV VARS (incluido
# GABOR_F1 con el valor autocalibrado que imprimió [GABOR]):
#     export GABOR_KERNEL=dir GABOR_FREQ=world GABOR_GAMMA=0.3 GABOR_F1=<valor>
#     python render.py -m $MODEL --skip_train --skip_mesh
#     python metrics.py -m $MODEL
# y fila al historial.
