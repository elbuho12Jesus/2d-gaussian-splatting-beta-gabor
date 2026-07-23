export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=45                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=5           # base run16 (óptimo del gate; >5 = más niebla, monótono)
CAP_MAX=7500000          # 7.5M = cap de run16 (óptimo honesto en flowers)
OPACITY_REG=0.01         # base run16 (techo perceptual; NO el 0.06 de run36)
SCALE_REG=0.01           # base run16

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run45: A/B AISLADO del JITTER de add_new_gs — réplica EXACTA de run16 (flowers) con el ÚNICO cambio en el
# CÓDIGO del jitter posicional de add_new_gs (scene/gaussian_model.py). NO cambia ningún flag/dial.
#
# QUÉ CAMBIÓ EN EL CÓDIGO (no en los flags): el jitter de add_new_gs desplazaba el clon en una dirección 3D
# ISÓTROPA aleatoria (torch.randn_like → esfera), lo que sacaba parte del clon FUERA del plano del surfel
# (por la normal). Ahora se desplaza a lo largo del EJE IN-PLANE DOMINANTE del surfel (marco propio, estilo
# densify_and_split), con signo ± aleatorio → el clon queda SOBRE el plano del surfel. Magnitud = escala del
# eje dominante × jitter_scale × err_src (sigue dirigido por error para sembrar huecos).
#
# PREGUNTA que responde: ¿mover los clones DENTRO del plano del surfel (como split) en vez de en 3D libre
# mejora el fondo (velo translúcido) sin dañar el foreground? Aísla SOLO la dirección del jitter.
#
# mcmc_jitter_scale = 1.5 (= run16, SIN cambiar). Es lo ADECUADO: mantenerlo aísla el efecto de la nueva
# dirección in-plane (si además tocara el dial, el test confundiría dos variables). La magnitud efectiva es
# comparable a run16 (antes: norma-2D de la escala en dirección 3D con ~2/3 en el plano; ahora: escala del
# eje dominante 100% en el plano), así que 1.5 sigue siendo un jitter razonable, no un salto de régimen.
#
# ÚNICO CAMBIO vs run16: la lógica del jitter en add_new_gs (código). Todos los dials = run16.
#
# OJOS EN ESTE RUN:
#   - Comparar honesto vs run16 (20.21 / 0.597 / 0.339) y vs run36 (21.16 / 0.5926 / 0.3555, mejor PSNR).
#   - Diagnóstico de brillo dmean(render−gt): run16 = +1.00. Si el jitter in-plane adelgaza el velo del fondo,
#     debería BAJAR (menos sobre-brillo translúcido). Vigilar también LPIPS (detalle fino del fondo).
#   - Cap: run16 topa exacto en 7.5M → confirmar N final = 7.5M para que sea A/B limpio.
#
# NO requiere recompilar el rasterizer (el jitter es Python puro en gaussian_model.py).
#
# Resto = run16: lambda_dist 10, dead_sustain 5, opacity_reg 0.01, scale_reg 0.01, ruido run9
# (--cov_noise --cov_noise_normal 1.0 --noise_lr 3e3), reset OFF, lambda_normal=0.05, opacity_cull=0.01,
# floater_cull_dist=0.2, mcmc_error_weight=3.5, mcmc_jitter_scale=1.5, iters=30000, densify_until=25000.
#
# Tras el run: render_server.sh (DATASET=flowers, RUN=45) + metrics.py. Comparar honesto-vs-honesto con
# run16/run36. Añadir fila al historial + docs/comparativa_runs.html.
python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --densify_mode mcmc \
    --iterations 30000 \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_until_iter 25000 \
    --lambda_normal 0.05 \
    --lambda_dist 10 \
    --opacity_reset_interval 1000000000 \
    --cap_max $CAP_MAX \
    --noise_lr 3e3 \
    --scale_reg $SCALE_REG \
    --opacity_reg $OPACITY_REG \
    --opacity_cull 0.01 \
    --floater_cull_dist 0.2 \
    --mcmc_error_weight 3.5 \
    --mcmc_jitter_scale 1.5 \
    --cov_noise \
    --cov_noise_normal 1.0 \
    --mcmc_dead_sustain $DEAD_SUSTAIN \
    2>&1 | tee $LOG
