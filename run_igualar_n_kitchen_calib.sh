#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000
export DEBUG_DENSIFY=1

# ═══════════════════════════════════════════════════════════════════════════
#  CALIBRACIÓN τ → N EN KITCHEN (paso 1 de 2 del experimento «igualar N»)
#
#  USO:   ./run_igualar_n_kitchen_calib.sh 1.5e-4              (un punto)
#         ./run_igualar_n_kitchen_calib.sh 1.7e-4 1.4e-4 1.1e-4  (barrido)
#
#  Corre runs CORTOS (15.000 iters) que solo sirven para medir cuántos splats
#  produce cada τ. No se miden métricas: el eval de estos runs NO significa nada
#  porque el modelo está a medio entrenar.
#
#  Cuando tengas el τ que da N ≈ 1.817.054, lanza el run de verdad:
#      ./run_igualar_n_kitchen.sh <tau>
#
#  Detalle, figuras y diseño del experimento: docs/igualar_n_atomo_vs_tent.html
# ═══════════════════════════════════════════════════════════════════════════
#
#  QUÉ SE PERSIGUE
#  ───────────────
#  El átomo converge en kitchen con un 10,4 % MENOS de splats que el tent
#  (run9: 1.628.932 contra los 1.817.054 de run78), así que el A/B átomo-vs-tent
#  lleva DOS deltas: el kernel y la capacidad. Con N igualado la comparación pasa
#  a ser vertical y la pérdida de 0,296 dB queda atribuida.
#
#  El dial es τ = --densify_grad_threshold: un splat se clona/divide cuando la
#  norma media de ∂L/∂(posición en pantalla) lo supera (gaussian_model.py:910 y
#  :827). Bajarlo hace que crucen más splats -> más densificación -> más N.
#
#  POR QUÉ 15.000 ITERACIONES Y NO 7.000
#  ─────────────────────────────────────
#  Porque 15.000 es densify_until_iter: ahí termina la densificación y N ya no se
#  mueve. Medido en los cinco runs átomo que tenemos:
#
#      log            N@~7000     N@~14700    N final    r(7k)    r(15k)
#      kitchen9      1.685.199   1.627.711   1.628.932   0.9666   1.0008
#      kitchen10     1.670.435   1.612.358   1.613.301   0.9658   1.0006
#      kitchen11     1.664.857   1.609.277   1.610.312   0.9672   1.0006
#      kitchen12     1.661.653   1.596.185   1.596.988   0.9611   1.0005
#      bonsai8       1.284.004   1.236.008   1.237.297   0.9636   1.0010
#
#  A 15 k el ratio es 1,0005-1,0010 (dispersión 0,05 %); a 7 k es 0,961-0,967
#  (dispersión 0,6 %, doce veces peor). Y el ratio de 7 k se midió TODO con
#  τ=2e-4: al bajar τ la densificación sigue añadiendo splats entre 7 k y 15 k,
#  así que ese ratio ni siquiera es transferible. 15 k cuesta ~25 min por punto
#  en vez de ~10, y a cambio la predicción es fiable.
#
#      N_final ≈ N@14700 × 1.0007
#
#  EL OBJETIVO
#  ───────────
#      N final buscado   = 1.817.054   (run78, el tent)
#      -> N@14700 buscado = 1.815.783
#      N@14700 actual (τ=2e-4, run9) = 1.627.711
#      => hace falta un +11,6 %
#
#  QUÉ τ PROBAR. No se puede predecir de la teoría: el histograma del gradiente
#  que imprime el log tiene la cola más corta que una lognormal (el ajuste
#  sobreestima el p99 en ×2,7), así que extrapolar la fracción que supera τ no es
#  fiable. Hay que medirlo. Un barrido razonable, en orden de apuesta:
#
#      1.5e-4   <- EMPEZAR AQUÍ (apuesta central)
#      1.7e-4   si 1.5e-4 se pasa de N
#      1.1e-4   si 1.5e-4 se queda corto
#
#  AVISO: bajar τ sube el consumo de memoria (más splats vivos). DEBUG_MEM=1000
#  está activo; si aparece OOM, sube τ. El colapso de run3 vino por el otro lado
#  (un prune agresivo), pero el riesgo de reventar la GPU por exceso de splats es
#  real y por eso la calibración va antes que el run largo.
# ───────────────────────────────────────────────────────────────────────────

if [ $# -eq 0 ]; then
    echo "Uso: $0 <tau> [tau2 tau3 ...]"
    echo
    echo "  Ej: $0 1.5e-4                 (un punto, ~25 min)"
    echo "      $0 1.7e-4 1.4e-4 1.1e-4   (barrido, ~75 min)"
    echo
    echo "  Objetivo: N@14700 ≈ 1.815.783  (=> N final ≈ 1.817.054, el tent run78)"
    echo "  Punto de partida: τ=2.0e-4 da N@14700 = 1.627.711  (hace falta +11,6 %)"
    exit 1
fi

DATASET=kitchen
ITERS=15000

LOCAL_DS=/home/jesus/Documents/Gaussian_splatting/360_extra_scenes/${DATASET}
if   [ -d "Datasets/${DATASET}" ]; then SOURCE="Datasets/${DATASET}"
elif [ -d "$LOCAL_DS" ];           then SOURCE="$LOCAL_DS"
else
    echo "⛔ No encuentro el dataset '${DATASET}'. Probado:"
    echo "   Datasets/${DATASET}   (servidor)"
    echo "   ${LOCAL_DS}   (local)"
    exit 1
fi

# ── Config = run9_gabor EXACTO. El único delta es τ, que va en el bucle ──────
export GABOR_KERNEL=dir
export GABOR_FREQ=world
export GABOR_GAMMA=0.3
export GABOR_KAPPA=1.0
export GABOR_F1=932.6041      # el autocalibrado de run9, explícito para que el
                              # log lo marque y render_server.sh lo lea igual
export SCALE_CLAMP_FACTOR=0.1
export SIZE_PRUNE_MODE=off

PRUNE_SUSTAIN=25
OPACITY_RESET_INTERVAL=3000
DENSIFY_FROM=500
DENSIFY_UNTIL=15000
DENSIFICATION_INTERVAL=100
PERCENT_DENSE=0.01
OPACITY_CULL=0.005
LAMBDA_DIST=0
LAMBDA_NORMAL=0.05
OPACITY_REG=0
SCALE_REG=0

OBJETIVO=1815783        # N@14700 que hace falta para acabar en 1.817.054
RESUMEN=()

for TAU in "$@"; do
    TAG=$(echo "$TAU" | tr -d '.-' | tr 'e' 'E')
    MODEL=output/m360/${DATASET}_calib_${TAG}
    LOG=logs/${DATASET}_calib_${TAG}.log
    echo
    echo "════════════════════════════════════════════════════════════════════"
    echo "  CALIBRACIÓN  τ = ${TAU}   (${ITERS} iters)   -> ${LOG}"
    echo "════════════════════════════════════════════════════════════════════"

    python train.py -s ${SOURCE} \
        -m $MODEL \
        --eval \
        --densify_mode classic \
        --iterations $ITERS \
        --test_iterations 2500 $ITERS \
        --densify_from_iter $DENSIFY_FROM \
        --densify_until_iter $DENSIFY_UNTIL \
        --densification_interval $DENSIFICATION_INTERVAL \
        --densify_grad_threshold $TAU \
        --percent_dense $PERCENT_DENSE \
        --opacity_reset_interval $OPACITY_RESET_INTERVAL \
        --opacity_cull $OPACITY_CULL \
        --lambda_normal $LAMBDA_NORMAL \
        --lambda_dist $LAMBDA_DIST \
        --opacity_reg $OPACITY_REG \
        --scale_reg $SCALE_REG \
        --classic_prune_sustain $PRUNE_SUSTAIN \
        2>&1 | tee $LOG

    # N de la última densificación (iter >= 14600)
    N=$(grep -oE '\[DENSIFY iter=1[45][0-9]{3}\][^|]*N=[0-9]+' $LOG | tail -1 | grep -oE 'N=[0-9]+' | cut -d= -f2)
    if [ -z "$N" ]; then
        RESUMEN+=("  τ=${TAU}   <sin dato: revisa ${LOG}>")
    else
        PRED=$(python -c "print(f'{int($N*1.0007):,d}'.replace(',','.'))")
        PCT=$(python -c "print(f'{100*($N/1627711-1):+.1f}')")
        DIFF=$(python -c "print(f'{100*($N/$OBJETIVO-1):+.1f}')")
        RESUMEN+=("  τ=${TAU}   N@14700=${N}   N_final≈${PRED}   (${PCT}% vs run9, ${DIFF}% vs objetivo)")
    fi

    # el ply de calibración no sirve para nada y ocupa ~400 MB
    if [ "${KEEP_PLY:-0}" != "1" ]; then
        rm -rf "${MODEL}/point_cloud"
        echo "  [limpieza] borrado ${MODEL}/point_cloud (KEEP_PLY=1 para conservarlo)"
    fi
done

echo
echo "════════════════════════════════════════════════════════════════════"
echo "  RESUMEN DE LA CALIBRACIÓN"
echo "  objetivo: N@14700 ≈ ${OBJETIVO}  (N final ≈ 1.817.054 = tent run78)"
echo "  ancla:    τ=2.0e-4 -> N@14700 = 1.627.711 (run9)"
echo "════════════════════════════════════════════════════════════════════"
printf '%s\n' "${RESUMEN[@]}"
echo
echo "  Cuando un τ caiga dentro de ±2 % del objetivo, lanza el run largo:"
echo "      ./run_igualar_n_kitchen.sh <tau>"
