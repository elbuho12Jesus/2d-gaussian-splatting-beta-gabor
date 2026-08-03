#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000
export DEBUG_NOISE=20

# ═══════════════════════════════════════════════════════════════════════════
#  EL KERNEL ÁTOMO CON DENSIFICACIÓN MCMC  (bonsai, run14 = tent / run15 = átomo)
#
#  USO:   ./run_mcmc_gabor_bonsai.sh tent      -> run14, el control
#         ./run_mcmc_gabor_bonsai.sh atomo     -> run15, el experimento
#
#  Los dos brazos salen de ESTE MISMO fichero y difieren en UNA línea: --a_lr.
#  Es la única forma de garantizar el delta único; dos scripts se desincronizan.
# ═══════════════════════════════════════════════════════════════════════════
#
#  POR QUÉ MCMC, Y POR QUÉ AHORA
#  ─────────────────────────────
#  Los 13 runs Gabor son TODOS clásicos. Y el A/B clásico arrastra siempre un segundo
#  delta: el átomo converge con ~10 % menos splats que el tent, así que "gana/pierde" y
#  "tiene menos primitivas" van juntos y no se pueden separar. run13 intentó igualar N
#  bajando tau y NO lo consiguió (se pasó un 40 %), porque tau->N no es predecible.
#
#  MCMC lo resuelve por construcción: cap_max SATURA. run44 terminó con Points=1500000
#  EXACTOS. Con el mismo cap en los dos brazos, N es idéntico por diseño y la
#  comparación es vertical de verdad. Es la opción B del §7 de
#  docs/igualar_n_atomo_vs_tent.html, la que el documento recomendaba desde el principio.
#
#  NO HACE FALTA TOCAR NADA DE CÓDIGO. Verificado el 2026-08-03:
#    - relocate_gs (gaussian_model.py:1159) y add_new_gs (:1238) YA propagan _a
#    - project_a_() vive en el bloque compartido del optimizer step (train.py:456),
#      fuera de la rama clásica -> la restricción a_n>=0, sum(a_n)<=cap se aplica igual
#    - sanitize_parameters repara _a con a_init_row()
#    - _reset_optimizer_state recorre TODOS los grupos, incluido el de "a"
#    - los diagnósticos [A] y [GABOR-W] están en el bloque de report, no en una rama
#  O sea: esto es un experimento de CONFIGURACIÓN, no un parche. La ruta clásica no se
#  toca, y `git status` solo debe enseñar este fichero nuevo.
#
#  POR QUÉ EL BRAZO TENT HAY QUE VOLVER A CORRERLO (y no vale run44)
#  ────────────────────────────────────────────────────────────────
#  run44 (bonsai, MCMC, 30.6665/0.9404/0.1862, cap 1.5M) es el mejor propio anterior en
#  bonsai, pero salió del OTRO repo (2d-gaussian-splatting-modificate) el 30/06, o sea
#  ANTES de los dos fixes del clamp de escala (26/07) y de los fixes de los prunes.
#  Compararse con él metería 3+ deltas a la vez. Su log está en
#      /home/jesus/Documents/Gaussian_splatting/2d-gaussian-splatting-modificate/logs/bonsai44.log
#  y vale como referencia histórica, NO como control.
#
#  CÓMO SE CONSTRUYE EL TENT AQUÍ (esto es lo bonito)
#  ──────────────────────────────────────────────────
#  En modo átomo el init es a=0, y a=0 => b=1 => S==1 => kernel = (1-r)^beta = EL TENT
#  EXACTO. Así que congelando _a con --a_lr 0 el brazo de control es el tent, generado
#  por el MISMO código, el mismo kernel CUDA y la misma ruta de datos que el experimento.
#  No es "parecido al tent": ES el tent, bit a bit, mientras a no se mueva.
#      OJO: GABOR_KAPPA=0 parece otra vía para lo mismo y NO LO ES: crashea. Con S=0 la
#      condición del simplex falla en j=1, rho sale -1 y el gather de project_a_ revienta
#      (gaussian_model.py:~230). No usar.
#
#  EL RIESGO ESPECÍFICO DE ESTA COMBINACIÓN (medirlo, no ignorarlo)
#  ───────────────────────────────────────────────────────────────
#  El kernel del átomo PICA POR ENCIMA DE 1 en el centro:
#      S(0) = b + sum(a_n) = 1 + gamma*sum(a_n)        (gamma=0.3)
#  Con los sum(a_n) medidos (0.24-0.30) son 1.07-1.09, y hasta 1.176 en el tope.
#  Y forward.cu:514 hace  alpha = min(0.99, opa * kernel), o sea que ese pico llega tal
#  cual a la opacidad efectiva.
#  En clásico da igual (el optimizador aprende una opacidad algo menor y ya). En MCMC NO
#  es tan inocuo: relocate_gs y add_new_gs aplican la regla de conservación de
#  transmitancia  new_alpha = 1-(1-alpha)^(1/(ratio+1))  sobre get_opacity, que asume que
#  la contribución del splat ES alpha. Con el átomo la contribución real es alpha*(1+
#  gamma*sum a), así que la regla queda sesgada un ~8 % — un sesgo que con el tent NO
#  existe (allí kernel(0)=1 exacto). No es un crash ni un showstopper, pero es
#  ESPECÍFICO de esta combinación y es lo primero que hay que mirar si el brazo átomo
#  sale inestable o con la opacidad derivando.
#
#  QUÉ MIRAR EN EL LOG
#   1. [GABOR] al arrancar: kernel=dir, freq=world, gamma=0.300, kappa=1.000,
#      sum(a_n) <= 0.5882, f1 autocalibrado = 204.3105 (el mismo de run8: la calibración
#      usa las escalas iniciales y el pcd es el mismo). DEBE salir igual en los DOS brazos.
#   2. Points=1500000 al final en los dos. Si un brazo no satura el cap, N no está
#      igualado y el experimento pierde su gracia -> decirlo al registrar.
#   3. [GABOR-W] en el brazo átomo. PREDICCIÓN FALSABLE: scale_reg=0.01 aprieta las
#      escalas y en modo world phi = f1*s_u/extent, así que f*W debería salir MÁS BAJO
#      que el 0.773 p50 del clásico run8 -> la onda más apagada. Si sale muy por debajo,
#      MCMC está silenciando el kernel y el empate no diría nada del átomo.
#   4. [A] el espectro. En bonsai clásico salía DECRECIENTE solo (a1>a2>a3 en todas las
#      iteraciones) y esa es la firma que correlaciona con ganar. ¿Se mantiene en MCMC?
#   5. VIOLACIONES neg/sum 0/0 y [CLAMP] recorte_max=0.000000 en los dos brazos.
#   6. En el brazo TENT: [A] debe dar sum(a_n) min/mean/max = 0.0000/0.0000/0.0000
#      SIEMPRE. Si se mueve, --a_lr 0 no está haciendo efecto y el control no es control.
#
#  BASELINES HONESTOS DE BONSAI (metrics.py, honesto-vs-honesto)
#      2DGS oficial                       31.36   / 0.9359 / 0.2042
#      run8_gabor  átomo, CLÁSICO         31.1348 / 0.9391 / 0.1852   N=1.237.297  <- mejor propio
#      run76       tent,  CLÁSICO         30.8194 / 0.9303 / 0.1971   N=1.374.712
#      run44       tent,  MCMC (otro repo)30.6665 / 0.9404 / 0.1862   N=1.500.000  <- referencia, NO control
#
#  LOS DESENLACES
#      átomo > tent con N EXACTAMENTE igual -> la victoria de bonsai queda ATRIBUIDA al
#                                              kernel, sin el asterisco del -10 % de N.
#                                              Es lo que le falta al resultado de run8.
#      empate                               -> el kernel necesita la libertad de N que le
#                                              da el clásico; MCMC lo neutraliza. Mirar
#                                              [GABOR-W] antes de concluir nada (punto 3).
#      átomo < tent                         -> MCMC y el átomo interfieren. El primer
#                                              sospechoso está escrito arriba: la regla de
#                                              transmitancia contra el pico kernel(0)>1.
# ───────────────────────────────────────────────────────────────────────────

BRAZO=${1:-}
case "$BRAZO" in
    tent)  RUN=14; A_LR=0.0    ;;   # _a congelado en 0 => S==1 => tent exacto
    atomo) RUN=15; A_LR=0.001  ;;   # el default de arguments/__init__.py:97
    *)
        echo "Uso: $0 tent|atomo"
        echo
        echo "  tent   -> run14, control: --a_lr 0, el kernel se queda en (1-r)^beta"
        echo "  atomo  -> run15, experimento: --a_lr 0.001, la forma se aprende"
        echo
        echo "  Lanza PRIMERO el tent: si no satura cap_max o el espectro no sale plano,"
        echo "  el control está mal y el experimento no se puede interpretar."
        exit 1 ;;
esac

DATASET=bonsai

LOCAL_DS=/home/jesus/Documents/Gaussian_splatting/360_extra_scenes/${DATASET}
if   [ -d "Datasets/${DATASET}" ]; then SOURCE="Datasets/${DATASET}"
elif [ -d "$LOCAL_DS" ];           then SOURCE="$LOCAL_DS"
else
    echo "⛔ No encuentro el dataset '${DATASET}'. Probado:"
    echo "   Datasets/${DATASET}   (servidor)"
    echo "   ${LOCAL_DS}   (local)"
    exit 1
fi

echo "════ run${RUN}: bonsai · MCMC · brazo ${BRAZO} (--a_lr ${A_LR}) ════"

# ── Kernel: IDÉNTICO en los dos brazos. El delta es a_lr, no el kernel ──────
export GABOR_KERNEL=dir     # legacy | radial | dir
export GABOR_FREQ=world     # norm | world
export GABOR_GAMMA=0.3      # pedestal. sum(a_n) <= 1/(2-gamma) = 0.5882
export GABOR_KAPPA=1.0      # 1.0 = sin acotar el contraste.  NO PONER 0: crashea (ver arriba)
# GABOR_F1 sin exportar -> autocalibrado por escena (bonsai: 204.3105, el de run8)

export SCALE_CLAMP_FACTOR=0.1
export SIZE_PRUNE_MODE=off  # medido: los dos prunes de tamaño restan (run3/run4)

# ── MCMC = receta de run44, que es el mejor MCMC medido en bonsai ───────────
CAP_MAX=1500000             # SATURA -> es lo que iguala N entre los dos brazos
DEAD_SUSTAIN=5
OPACITY_REG=0.01
SCALE_REG=0.01
NOISE_LR=3e3
MCMC_ERROR_WEIGHT=3.5
MCMC_JITTER_SCALE=1.5
OPACITY_CULL=0.01
FLOATER_CULL_DIST=0.2
COV_NOISE_NORMAL=1.0
LAMBDA_DIST=10
LAMBDA_NORMAL=0.05
ITERATIONS=30000
DENSIFY_UNTIL=25000

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

python train.py -s ${SOURCE} \
    -m $MODEL \
    --eval \
    --densify_mode mcmc \
    --iterations $ITERATIONS \
    --test_iterations 2500 7000 15000 20000 25000 30000 \
    --densify_until_iter $DENSIFY_UNTIL \
    --opacity_reset_interval 1000000000 \
    --cap_max $CAP_MAX \
    --a_lr $A_LR \
    --noise_lr $NOISE_LR \
    --scale_reg $SCALE_REG \
    --opacity_reg $OPACITY_REG \
    --opacity_cull $OPACITY_CULL \
    --floater_cull_dist $FLOATER_CULL_DIST \
    --mcmc_error_weight $MCMC_ERROR_WEIGHT \
    --mcmc_jitter_scale $MCMC_JITTER_SCALE \
    --cov_noise \
    --cov_noise_normal $COV_NOISE_NORMAL \
    --mcmc_dead_sustain $DEAD_SUSTAIN \
    --lambda_normal $LAMBDA_NORMAL \
    --lambda_dist $LAMBDA_DIST \
    2>&1 | tee $LOG

# ── Después ────────────────────────────────────────────────────────────────
#     ./render_server.sh ${RUN} 30000 ${DATASET}
#     python metrics.py -m output/m360/${DATASET}_beta_run${RUN}
#
# render_server.sh lee las env vars del log del train y verifica que el kernel del render
# coincide. En el brazo TENT esto es redundante (con a=0 el kernel no depende de f1 ni de
# gamma), pero se hace igual: si algún día a deja de ser 0, el aviso ya está puesto.
#
# AL REGISTRAR EN historial_runs.csv: decir que el brazo tent es "--a_lr 0 sobre kernel
# átomo", NO "kernel tent" a secas — son el mismo kernel matemáticamente, pero por una ruta
# de código distinta a la de run76, y eso hay que poder reconstruirlo dentro de seis meses.
