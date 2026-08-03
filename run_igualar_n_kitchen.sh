#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000
export DEBUG_DENSIFY=1

# ═══════════════════════════════════════════════════════════════════════════
#  IGUALAR N: EL ÁTOMO CON LA MISMA CAPACIDAD QUE EL TENT  (run13, kitchen)
#
#  USO:   ./run_igualar_n_kitchen.sh 1.5e-4      <- el τ que salga de la calibración
#
#  ANTES hay que calibrar τ, o el run no responde la pregunta:
#      ./run_igualar_n_kitchen_calib.sh 1.5e-4
#
#  Detalle, figuras y los tres desenlaces: docs/igualar_n_atomo_vs_tent.html
# ═══════════════════════════════════════════════════════════════════════════
#
#  = run9_gabor EXACTO + UN SOLO DELTA: --densify_grad_threshold.
#
#  LA PREGUNTA
#  ───────────
#  run9 (átomo) pierde 0,296 dB contra run78 (tent) en kitchen, pero converge con
#  un 10,4 % MENOS de splats (1.628.932 contra 1.817.054). Las dos lecturas
#  encajan con ese dato y son incompatibles entre sí:
#
#      «es el kernel»    -> el átomo representa peor esta escena
#      «es la capacidad» -> el átomo no es peor, juega con un 10 % menos de
#                           primitivas porque su propio acierto le cierra la
#                           densificación (menos residuo -> menos gradiente de
#                           posición -> menos splats cruzan τ)
#
#  Con N igualado la comparación pasa a ser vertical y queda atribuida.
#
#  POR QUÉ ESTO NO ES «UN A/B DE UN SOLO DELTA» (y hay que decirlo al registrarlo)
#  ──────────────────────────────────────────────────────────────────────────────
#  Igualar N cuesta DESIGUALAR τ: el brazo tent (run78) corrió con τ=2e-4 y este
#  corre con τ menor. No se elimina el segundo delta, se CAMBIA POR OTRO. Se
#  prefiere así porque N es una propiedad del MODELO que se compara, mientras que
#  τ es del PROCESO que lo produjo: dos modelos con la misma capacidad son
#  comparables aunque se hayan construido con umbrales distintos, y al revés no.
#
#  QUÉ MIRAR EN EL LOG
#   1. [GABOR] al arrancar: kernel=dir, freq=world, gamma=0.300, f1=932.604
#      "<- env GABOR_F1", kappa=1.000, sum(a_n) <= 0.5882. Igual que run9.
#   2. [DENSIFY iter=14700] N= : tiene que caer cerca de 1.815.783. Si se desvía
#      más de un ±3 %, el run sigue valiendo pero hay que decir el N real al
#      comparar — la pregunta era «a igual N», no «a N parecido».
#   3. El p90 de grad_pos contra el nuevo τ. En run9 el p90 acababa 3,5 veces por
#      debajo del umbral (5,7e-5 contra 2,0e-4) = la densificación se apagaba
#      sola. Con τ menor esa distancia se acorta; si el p90 acaba POR ENCIMA del
#      nuevo τ, la densificación no se ha saturado y N depende de dónde se corte,
#      no del equilibrio -> el resultado sería frágil.
#   4. [CLAMP] recorte_max=0.000000 y VIOLACIONES neg/sum 0/0.
#   5. [A] espectro: en run9 salía decreciente (a1 0,1156 > a2 0,0829 > a3 0,0733)
#      y ES LA FIRMA BUENA (la de bonsai, que gana). Si al densificar más se
#      aplana o se invierte, ojo: en el sweep de f1 los cuatro puntos ordenaban la
#      métrica por lo plano que quedaba el espectro.
#   6. [BETA] mean ~2,30 en run9. Subir beta apaga la onda.
#   7. Memoria: bajar τ significa más splats vivos. DEBUG_MEM=1000 activo.
#
#  BASELINES HONESTOS DE KITCHEN (metrics.py, honesto-vs-honesto)
#      run78  tent, clásico limpio, τ=2e-4   30.2876 / 0.9266 / 0.1255  N=1.817.054  <- A BATIR
#      run9   átomo, τ=2e-4                  29.9916 / 0.9248 / 0.1259  N=1.628.932  <- EL A/B
#      run10  átomo, f1÷2, τ=2e-4            29.9973 / 0.9253 / 0.1249  N=1.613.301
#      2DGS oficial                          30.3389 / 0.9210 / 0.1383
#  Y la referencia de que el átomo SÍ puede ganar:
#      bonsai run8 31.1348/0.9391/0.1852 vs tent run76 30.8194/0.9303/0.1971
#
#  LOS TRES DESENLACES
#      recupera los ~0,29 dB  -> era CAPACIDAD. τ está calibrado para otro kernel;
#                                es un problema de integración, no del kernel. Y
#                                hay que revisar bonsai, donde la victoria podría
#                                crecer.
#      no recupera nada       -> era el KERNEL. Cierra la vía de kitchen y refuerza
#                                la hipótesis de escena (superficie lisa y
#                                especular = no hay textura oscilatoria).
#      recupera una parte     -> suman las dos. Cuantifica cuánto es de cada causa
#                                y da el primer punto real de la curva calidad-vs-N
#                                del átomo.
#  Los tres cambian la decisión siguiente, que es lo que hace que el run valga.
# ───────────────────────────────────────────────────────────────────────────

TAU=${1:-}
if [ -z "$TAU" ]; then
    echo "Uso: $0 <densify_grad_threshold>"
    echo
    echo "  El argumento es obligatorio: es EL delta del experimento y sale de"
    echo "  la calibración, no de una corazonada."
    echo
    echo "      ./run_igualar_n_kitchen_calib.sh 1.5e-4"
    echo
    echo "  Objetivo: N final ≈ 1.817.054 (= run78, el tent)."
    echo "  Referencia: τ=2.0e-4 da 1.628.932 (run9), un 10,4 % menos."
    exit 1
fi

RUN=13
DATASET=kitchen

LOCAL_DS=/home/jesus/Documents/Gaussian_splatting/360_extra_scenes/${DATASET}
if   [ -d "Datasets/${DATASET}" ]; then SOURCE="Datasets/${DATASET}"
elif [ -d "$LOCAL_DS" ];           then SOURCE="$LOCAL_DS"
else
    echo "⛔ No encuentro el dataset '${DATASET}'. Probado:"
    echo "   Datasets/${DATASET}   (servidor)"
    echo "   ${LOCAL_DS}   (local)"
    exit 1
fi

echo "════ run${RUN}: kitchen · átomo con N igualado al tent · τ=${TAU} (run9 usaba 2.0e-4) ════"

# ── Config = run9_gabor EXACTO ───────────────────────────────────────────────
export GABOR_KERNEL=dir     # legacy | radial | dir
export GABOR_FREQ=world     # norm | world
export GABOR_GAMMA=0.3      # pedestal, = AdaGaR. sum(a_n) <= 1/(2-gamma) = 0.5882
export GABOR_KAPPA=1.0      # 1.0 = sin acotar el contraste
export GABOR_F1=932.6041    # el autocalibrado de run9, explícito y determinista

export SCALE_CLAMP_FACTOR=0.1
export SIZE_PRUNE_MODE=off  # medido: los dos prunes de tamaño restan (run3/run4)

PRUNE_SUSTAIN=25
OPACITY_RESET_INTERVAL=3000
DENSIFY_FROM=500
DENSIFY_UNTIL=15000
DENSIFICATION_INTERVAL=100
DENSIFY_GRAD_THRESHOLD=$TAU   # ← EL DELTA (run9: 0.0002)
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

python train.py -s ${SOURCE} \
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
#     ./render_server.sh ${RUN} 30000 ${DATASET}
#     python metrics.py -m output/m360/${DATASET}_beta_run${RUN}
#
# render_server.sh lee las env vars del log del train y verifica que el kernel del
# render coincide (GABOR_F1 incluido). OJO: metrics.py solo lee test/; el vídeo
# vive en traj/ y solo se regenera con --render_path. render_server.sh hace las dos.
#
# Si el train corre en el servidor y el render en local, bájate ANTES el log:
#     cd logs && ./cpsh.sh ${RUN} ${DATASET}
# y kitchen NO está en local, así que el render de esta escena va en el servidor.
#
# AL REGISTRAR LA FILA EN historial_runs.csv: decir el N REAL obtenido y que este
# run lleva τ distinto del tent. No es un A/B de un solo delta y no debe leerse
# como tal (ver la nota de arriba).
