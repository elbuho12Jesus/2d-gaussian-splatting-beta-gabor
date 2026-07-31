#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000
export DEBUG_DENSIFY=1

# ═══════════════════════════════════════════════════════════════════════════
#  EL KERNEL ÁTOMO FUERA DE FLOWERS — bonsai y kitchen, un script, dos brazos
#
#  USO:   ./run_gabor_escenas.sh bonsai    -> run8, output/m360/bonsai_beta_run8
#         ./run_gabor_escenas.sh kitchen   -> run9, output/m360/kitchen_beta_run9
#
#  ESTADO: los dos SIN LANZAR (creados el 2026-07-31).
# ═══════════════════════════════════════════════════════════════════════════
#
#  = run5_gabor EXACTO (kernel átomo, dir+world, γ=0.3, κ=1.0) + UN SOLO DELTA:
#    LA ESCENA. Ni un hiperparámetro más se mueve.
#
#  POR QUÉ (medido, sweep de GABOR_F1 cerrado el 2026-07-31):
#  en flowers el átomo EMPATA con el tent (run5 20.6898 vs run67 20.6684) y el sweep
#  de f1 demostró que no es porque la onda esté apagada: con f1 ×4 (run7) se encendió
#  5,9× más splats (apagados 93,40 % -> 64,12 %), con MÁS amplitud (Σa 0,2909 ->
#  0,3429) y MENOS kernels planos (21,4 % -> 12,9 %) ... y las métricas no se movieron
#  (20.6898 / 20.5999 / 20.7167 = 0,12 dB de dispersión). O sea: en flowers LA ONDA NO
#  COMPRA NADA. Todo el +0,53 dB del átomo sobre el legacy vino de la ENVOLVENTE.
#
#  LA PREGUNTA QUE RESPONDE ESTE RUN:  ¿eso es una propiedad de LA ESCENA o DEL KERNEL?
#    · flowers es 360 exterior: césped y follaje = textura estocástica de alta
#      frecuencia SIN estructura oscilatoria coherente. Un armónico aprendido no tiene
#      nada que "encajar" ahí.
#    · bonsai y kitchen son INTERIORES ACOTADAS, con superficies planas, bordes rectos,
#      rejillas y texturas repetitivas (baldosas, listones, tramas). Ahí una onda SÍ
#      tiene algo que representar.
#  Si el átomo tampoco suma aquí, el veredicto pasa de "en flowers no" a "el kernel
#  oscilatorio no aporta", y la línea del Gabor queda cerrada de verdad.
#
#  POR QUÉ ESTAS DOS ESCENAS Y NO OTRAS: son las únicas con el baseline del kernel ya
#  medido con ESTA MISMA config clásica limpia (run76 y run78) -> el A/B átomo-vs-tent
#  es directo, honesto-vs-honesto y sin gastar un run en el baseline.
#
#  ⚠ GABOR_F1 NO SE EXPORTA A PROPÓSITO. Se autocalibra por escena (f·W=1 en el p90 de
#    las escalas iniciales, gaussian_model.py:376), y el valor de flowers (46.6247)
#    NO sirve aquí: depende de extent y de la nube SfM inicial, que son otras. El log
#    imprimirá "[GABOR] f1 autocalibrado = X rad/extent" — ese es el número de la
#    escena. Sin sweep: en flowers ya se midió que f1 no es palanca (×1/×2/×4 planos).
#
#  QUÉ MIRAR EN EL LOG
#   1. [GABOR] al arrancar: kernel=dir | freq=world | gamma=0.300 | kappa=1.000 y
#      f1=0 <- autocalibrado, seguido de la línea "f1 autocalibrado = X". ANOTAR X:
#      hace falta para el render (render_server.sh la lee sola del log, pero si el
#      render se hace en otra máquina hay que copiarla al bloque OVERRIDE).
#   2. [GABOR-W] en cada test_iteration: % de APAGADOS (f·W<0,5). En flowers arrancaba
#      en 85,8 % @2500 y subía al densificar. Si aquí sale MUCHO más bajo, la onda se
#      está usando de verdad y el A/B mide algo nuevo; si sale igual de alto, la
#      autocalibración vuelve a quedarse corta por la misma razón (usa las escalas
#      INICIALES y la densificación encoge los surfels).
#   3. [A] la firma que en flowers apareció en run4/run5/run6/run7 SIN excepción:
#      a1 BAJA y a3 SUBE durante el run, hasta a3 > a1 a 30 k (espectro CRECIENTE) y
#      Σa<0,05 subiendo. Si en una escena con estructura esto NO pasa —si a1 se queda
#      arriba— entonces el espectro creciente era la escena, no el kernel, y la palanca
#      siguiente (empujar a1>a2>a3) queda justificada.
#   4. [BETA] mean: en los clásicos sanos de estas escenas 2,59-2,90. Subir beta apaga
#      la onda (W = 2·s·σ_env(β)): es la vía de escape barata del optimizador.
#   5. [CLAMP] recorte_max = 0.000000 y VIOLACIONES neg/sum 0/0. Si no, el run no vale.
#   6. N de splats: bonsai/kitchen son acotadas y el clásico creció a 1,37 M / 1,82 M
#      con el tent. Un salto grande respecto a eso es señal, no ruido.
#
#  BASELINES HONESTOS (metrics.py, honesto-vs-honesto, mismas vistas)
#    BONSAI                                   PSNR / SSIM / LPIPS
#      run76  clásico limpio, kernel TENT     30.8194 / 0.9303 / 0.1971  <- EL A/B DIRECTO
#      2DGS oficial                           31.3600 / 0.9359 / 0.2042
#      run44  (MCMC, mejor propio en bonsai)  30.6665 / 0.9404 / 0.1862
#    KITCHEN
#      run78  clásico limpio, kernel TENT     30.2876 / 0.9266 / 0.1255  <- EL A/B DIRECTO
#      2DGS oficial                           30.3389 / 0.9210 / 0.1383
# ───────────────────────────────────────────────────────────────────────────

ESCENA=${1:-}
case "$ESCENA" in
  bonsai)  DATASET=bonsai;  RUN=8 ;;
  kitchen) DATASET=kitchen; RUN=9 ;;
  *)
    echo "Uso: $0 <bonsai|kitchen>"
    echo
    echo "  bonsai   -> run8   (baseline tent run76: 30.8194/0.9303/0.1971)"
    echo "  kitchen  -> run9   (baseline tent run78: 30.2876/0.9266/0.1255)"
    echo
    echo "La escena es obligatoria: el número de run va atado a ella."
    exit 1 ;;
esac

# ── Dónde está el dataset (mismo criterio que render_server.sh) ────────────
# En el servidor es Datasets/<escena>; en local vive fuera del repo.
LOCAL_DS=/home/jesus/Documents/Gaussian_splatting/360_extra_scenes/${DATASET}
if   [ -d "Datasets/${DATASET}" ]; then SOURCE="Datasets/${DATASET}"
elif [ -d "$LOCAL_DS" ];           then SOURCE="$LOCAL_DS"
else
    echo "⛔ No encuentro el dataset '${DATASET}'. Probado:"
    echo "   Datasets/${DATASET}   (servidor)"
    echo "   ${LOCAL_DS}   (local)"
    exit 1
fi
echo "════ run${RUN} · ${DATASET} · kernel ÁTOMO · dataset ${SOURCE} ════"

# ── El kernel: IDÉNTICO a run5_gabor. GABOR_F1 se autocalibra (ver arriba) ──
export GABOR_KERNEL=dir     # legacy | radial | dir
export GABOR_FREQ=world     # norm | world
export GABOR_GAMMA=0.3      # pedestal, = AdaGaR. sum(a_n) <= 1/(2-gamma) = 0.5882
export GABOR_KAPPA=1.0      # 1.0 = sin acotar el contraste

# ── Todo lo demás = run5_gabor = run76/run78 (train_CLASSIC_server.sh) ─────
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
# El render lee las env vars del log del train y verifica que coinciden, f1
# autocalibrado incluido (render_server.sh contempla los dos casos de f1):
#
#     ./render_server.sh ${RUN} 30000 ${DATASET}
#     python metrics.py -m output/m360/${DATASET}_beta_run${RUN}
#
# y luego la fila al historial. OJO: metrics.py solo lee test/; el vídeo vive en
# traj/ y solo se regenera con --render_path. render_server.sh hace las dos.
#
# Si el train corre en el servidor y el render en local, bájate ANTES el log:
#     cd logs && ./cpsh.sh ${RUN} ${DATASET}
# sin él render_server.sh aborta (correr con los defaults = el bug de run5).
