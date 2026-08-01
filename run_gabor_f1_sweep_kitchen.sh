#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000
export DEBUG_DENSIFY=1

# ═══════════════════════════════════════════════════════════════════════════
#  SWEEP DE GABOR_F1 EN KITCHEN — A LA BAJA. Un script, tres brazos.
#
#  USO:   ./run_gabor_f1_sweep_kitchen.sh 2   -> run10, f1 = 466.3021  (÷2)
#         ./run_gabor_f1_sweep_kitchen.sh 4   -> run11, f1 = 233.1510  (÷4)
#         ./run_gabor_f1_sweep_kitchen.sh 8   -> run12, f1 = 116.5755  (÷8)
#
#  ORDEN RECOMENDADO: lanzar ÷2 PRIMERO. Es el brazo con la predicción concreta
#  (ver abajo); ÷4 solo si ÷2 mejora pero no basta, y ÷8 solo si la curva sigue
#  subiendo. NO hace falta correr los tres.
#
#  ESTADO: los tres SIN LANZAR (creados el 2026-08-02).
# ═══════════════════════════════════════════════════════════════════════════
#
#  = run9_gabor EXACTO + UN SOLO DELTA: GABOR_F1. Kernel átomo dir+world,
#    gamma 0.3, kappa 1.0, resto idéntico a run78 (clásico limpio).
#
#  POR QUÉ SE BAJA f1 (y no se sube, como en flowers)
#  ─────────────────────────────────────────────────
#  run9 (kitchen, f1 autocalibrado = 932.6041) dio 29.9916/0.9248/0.1259 contra
#  30.2876/0.9266/0.1255 del tent (run78): -0,296 dB, empate en SSIM y LPIPS.
#  Pierde, pero NO por lo mismo que flowers. En flowers la onda estaba APAGADA
#  (93,4 % de splats con f·W<0,5). En kitchen está DEMASIADO ENCENDIDA:
#
#      f·W p50 = 1,600   p90 = 7,224   apagados 13,5 %   encendidos 65,5 %
#
#  Un p90 de 7,2 significa que uno de cada diez splats tiene MÁS DE SIETE
#  anchuras de onda dentro de su propia huella: el kernel oscila más rápido que
#  el detalle que hay que representar = ruido de alta frecuencia. Es el fallo
#  que run_gabor_f1_sweep.sh ya anticipaba por escrito ("si apagados baja y el
#  eval BAJA, f1 se pasó"). Y la forma es la BUENA: el espectro sale decreciente
#  (a1 0,1156 > a2 0,0829 > a3 0,0733), igual que en bonsai, que gana. Lo que
#  sobra es frecuencia, no amplitud ni monotonía.
#
#  De dónde salió un f1 tan alto: la autocalibración fija f·W=1 en el p90 de las
#  escalas INICIALES, y los surfels de kitchen nacen diminutos (s_p90 = 0,01886
#  contra 0,08963 en bonsai y 0,36666 en flowers) -> f1 = 932,6, veinte veces el
#  de flowers. Luego los surfels CRECEN durante el run (en un interior acotado
#  crecen; en un 360 exterior encogen) y f·W se dispara.
#
#  LA PREDICCIÓN CONCRETA DE ÷2  <-- esto es lo que hace que valga la pena
#  ─────────────────────────────────────────────────────────────────────
#  bonsai (run8) GANA en las tres, y su histograma es:
#
#                        f·W p50    f·W p90   apagados  encendidos
#      bonsai run8  ->     0,773      3,468     34,0 %     40,7 %   (GANA)
#      kitchen run9 ->     1,600      7,224     13,5 %     65,5 %   (pierde)
#      kitchen ÷2   ->    ~0,800     ~3,610       ?          ?      <- AQUÍ
#
#  Dividir f1 entre 2 coloca a kitchen PRÁCTICAMENTE EN EL PUNTO DE BONSAI. Si
#  el régimen de f·W es lo que separa ganar de perder, ÷2 debería recuperar los
#  0,296 dB y quedar por encima del tent. Si NO los recupera, entonces el
#  régimen de f·W no es la explicación y hay que buscar en otro sitio (kitchen
#  tiene mucha superficie especular y lisa: puede que simplemente no haya
#  textura oscilatoria que representar).
#
#  Es una predicción falsable con un solo run, que es justo lo que se quiere.
#
#  QUÉ MIRAR EN EL LOG
#   1. [GABOR] al arrancar: f1 = el valor de abajo y "<- env GABOR_F1" (NO
#      "autocalibrado"). Con la env var puesta la autocalibración NO corre.
#   2. [GABOR-W]: el objetivo NO es bajar los apagados —aquí ya son bajos— sino
#      SUBIRLOS hasta el entorno de bonsai (~30-35 %) y meter el p90 por debajo
#      de ~4. Se mira el histograma, nunca el máximo.
#      · eval SUBE y p90 baja  -> confirmado: el problema era la calibración.
#      · eval sigue igual      -> f·W no era la explicación; parar el sweep.
#      · eval BAJA             -> nos hemos pasado al otro lado (onda apagada);
#                                 el óptimo está entre 932,6 y este valor.
#   3. [A] el espectro debe seguir DECRECIENTE (a1 > a2 > a3). Si al bajar f1 se
#      invierte (a3 > a1, la firma de flowers), es señal de que el kernel se
#      queda sin nada que representar y se va al régimen malo.
#   4. [BETA] mean ~2,30 en run9. Subir beta apaga la onda: vigilar que no se
#      dispare y enmascare el efecto de f1.
#   5. [CLAMP] recorte_max = 0.000000 y VIOLACIONES neg/sum 0/0.
#   6. N de splats: run9 dio 1.628.932 (10,4 % MENOS que el tent de run78).
#
#  BASELINES HONESTOS DE KITCHEN (metrics.py, honesto-vs-honesto)
#      run78  clásico limpio, kernel TENT   30.2876 / 0.9266 / 0.1255  <- A BATIR
#      2DGS oficial                         30.3389 / 0.9210 / 0.1383
#      run9   átomo, f1 = 932.6041 (1×)     29.9916 / 0.9248 / 0.1259  <- EL A/B
#  Y la referencia de que el átomo SÍ puede ganar:
#      bonsai run8 31.1348/0.9391/0.1852 vs tent run76 30.8194/0.9303/0.1971
# ───────────────────────────────────────────────────────────────────────────

DIV=${1:-}
case "$DIV" in
  2) RUN=10; export GABOR_F1=466.3021 ;;   # 932.6041 / 2
  4) RUN=11; export GABOR_F1=233.1510 ;;   # 932.6041 / 4
  8) RUN=12; export GABOR_F1=116.5755 ;;   # 932.6041 / 8
  *)
    echo "Uso: $0 <2|4|8>   (DIVISOR de f1 sobre el autocalibrado 932.6041 de run9)"
    echo
    echo "  2  -> run10, GABOR_F1=466.3021   [EMPEZAR POR AQUI: deja f*W p90 ~3,6"
    echo "                                    = el punto de bonsai, que gana]"
    echo "  4  -> run11, GABOR_F1=233.1510   [solo si ÷2 mejora pero no basta]"
    echo "  8  -> run12, GABOR_F1=116.5755   [solo si la curva sigue subiendo]"
    echo
    echo "El argumento es obligatorio: sin el se relanzaria un brazo por descuido."
    exit 1 ;;
esac
echo "════ run${RUN}: kitchen · GABOR_F1=${GABOR_F1} (÷${DIV} del autocalibrado de run9) ════"

DATASET=kitchen

# ── Dónde está el dataset (mismo criterio que render_server.sh) ────────────
LOCAL_DS=/home/jesus/Documents/Gaussian_splatting/360_extra_scenes/${DATASET}
if   [ -d "Datasets/${DATASET}" ]; then SOURCE="Datasets/${DATASET}"
elif [ -d "$LOCAL_DS" ];           then SOURCE="$LOCAL_DS"
else
    echo "⛔ No encuentro el dataset '${DATASET}'. Probado:"
    echo "   Datasets/${DATASET}   (servidor)"
    echo "   ${LOCAL_DS}   (local)"
    exit 1
fi

# ── EL DELTA: solo f1 (arriba). Todo lo demás = run9_gabor ────────────────
export GABOR_KERNEL=dir     # legacy | radial | dir
export GABOR_FREQ=world     # norm | world
export GABOR_GAMMA=0.3      # pedestal, = AdaGaR. sum(a_n) <= 1/(2-gamma) = 0.5882
export GABOR_KAPPA=1.0      # 1.0 = sin acotar el contraste

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
# render_server.sh lee las env vars del log del train y verifica que el kernel
# del render coincide con el del train (GABOR_F1 incluido):
#
#     ./render_server.sh ${RUN} 30000 ${DATASET}
#     python metrics.py -m output/m360/${DATASET}_beta_run${RUN}
#
# y luego la fila al historial. OJO: metrics.py solo lee test/; el vídeo vive en
# traj/ y solo se regenera con --render_path. render_server.sh hace las dos.
#
# Si el train corre en el servidor y el render en local, bájate ANTES el log:
#     cd logs && ./cpsh.sh ${RUN} ${DATASET}
# sin él render_server.sh aborta. Y kitchen NO está en local, así que el render
# de esta escena va en el servidor salvo que te bajes el dataset.
