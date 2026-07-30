#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ═══════════════════════════════════════════════════════════════════════════
#  RENDER + verificación de consistencia train ↔ render
# ═══════════════════════════════════════════════════════════════════════════
#
#  POR QUÉ ESTE SCRIPT LEE EL LOG DEL TRAIN (2026-07-30, le costó un render a run5):
#  el modelo NO guarda la configuración del kernel. Vive en env vars, y render.py
#  hace load_ply (no create_from_pcd), así que:
#    · la AUTOCALIBRACIÓN de GABOR_F1 no se ejecuta -> sin la env var queda en 0;
#    · si la cota de Σaₙ del render es más estrecha que la del train (legacy 0.5 vs
#      átomo 0.5882), load_ply REPROYECTA los coeficientes y MUTA el modelo.
#  El primer render de run5 salió con kernel=legacy/freq=norm/f1=0/Σa≤0.5 sobre un
#  modelo entrenado con dir/world/46.6247/0.5882: 29.2% de los splats reproyectados
#  y 15.6151/0.3050/0.5591 = −5.07 dB, con dianas en el césped y aspecto de óleo.
#  El modelo estaba perfecto: 20.6898 al re-renderizarlo bien.
#
#  Por eso aquí NO se escriben las env vars a mano: se EXTRAEN del log del train, y
#  al final se COMPARA la línea [GABOR] del train contra la de cada render.
#
#  Si no tienes el log del train (p.ej. el modelo viene de otra máquina), rellena el
#  bloque OVERRIDE y el script lo respetará. Sin log y sin OVERRIDE, ABORTA: correr
#  con los defaults es exactamente el bug de run5.
# ───────────────────────────────────────────────────────────────────────────

# ── ÚNICO bloque a editar entre runs ───────────────────────────────────────
DATASET=flowers          # carpeta en Datasets/
RUN=5                    # número de run
ITER=30000               # iteración (checkpoint) a renderizar

MODEL=output/m360/${DATASET}_beta_run${RUN}
TRAIN_LOG=logs/${DATASET}${RUN}.log      # el log que dejó train.py

# ── OVERRIDE opcional: descomenta SOLO si no tienes el log del train ───────
# export GABOR_KERNEL=dir GABOR_FREQ=world GABOR_GAMMA=0.3 GABOR_KAPPA=1.0
# export GABOR_F1=46.6247
# export SCALE_CLAMP_FACTOR=0.1 SIZE_PRUNE_MODE=off WS_PRUNE_FRACTION=0.7
# ───────────────────────────────────────────────────────────────────────────

set -u

# ═══ 1) Derivar la configuración del log del train ═════════════════════════
if [ -f "$TRAIN_LOG" ]; then
    echo "════ Leyendo configuración de $TRAIN_LOG ════"

    # Exporta $1=NOMBRE solo si el valor $2 no viene vacío y la var no está ya puesta.
    # Exportar VACÍO sería peor que no exportar: el código Python trata "" como
    # "no definida" y cae al default (legacy/norm/f1=0), o sea el bug de run5 otra vez,
    # pero encima con el script diciendo que había leído la configuración.
    set_if() { [ -n "${!1:-}" ] && return 0; [ -z "$2" ] && return 0; export "$1=$2"; }

    # [GABOR] kernel=dir (mode=2) | freq=world | gamma=0.300 | f1=0 <- ... | kappa=1.000 | ...
    G_LINE=$(grep -am1 '^\[GABOR\] kernel=' "$TRAIN_LOG" || true)
    if [ -n "$G_LINE" ]; then
        set_if GABOR_KERNEL "$(sed -n 's/.*kernel=\([a-z]*\) .*/\1/p' <<<"$G_LINE")"
        set_if GABOR_FREQ   "$(sed -n 's/.*| freq=\([a-z]*\) .*/\1/p' <<<"$G_LINE")"
        set_if GABOR_GAMMA  "$(sed -n 's/.*| gamma=\([0-9.]*\) .*/\1/p' <<<"$G_LINE")"
        set_if GABOR_KAPPA  "$(sed -n 's/.*| kappa=\([0-9.]*\) .*/\1/p' <<<"$G_LINE")"

        # f1: OJO — la línea de arranque imprime f1=0 cuando se autocalibra. El valor
        # real sale DESPUÉS: "[GABOR] f1 autocalibrado = 46.6247 rad/extent". Si f1 vino
        # por env var no hay tal línea y el bueno es el de la línea de arranque.
        if [ -z "${GABOR_F1:-}" ]; then
            F1=$(sed -n 's/.*\[GABOR\] f1 autocalibrado = \([0-9.]*\) rad.*/\1/p' "$TRAIN_LOG" | head -1)
            [ -z "$F1" ] && F1=$(sed -n 's/.*| f1=\([0-9.]*\) <-.*/\1/p' <<<"$G_LINE")
            set_if GABOR_F1 "$F1"
        fi
    else
        echo "⚠  $TRAIN_LOG no tiene línea [GABOR] (run anterior al kernel átomo)."
        echo "   Se asumen los defaults del código: kernel=legacy, freq=norm, f1=0."
        echo "   La verificación de kernel queda SIN COBERTURA para este run."
    fi

    # [CLAMP] scale_clamp_factor = 0.1000
    # [PRUNE-TAM] size_prune_mode = off (...) | ws_prune_fraction = 0.70
    set_if SCALE_CLAMP_FACTOR "$(sed -n 's/.*\[CLAMP\] scale_clamp_factor = \([0-9.]*\).*/\1/p' "$TRAIN_LOG" | head -1)"
    set_if SIZE_PRUNE_MODE    "$(sed -n 's/.*size_prune_mode = \([a-z]*\) .*/\1/p' "$TRAIN_LOG" | head -1)"
    set_if WS_PRUNE_FRACTION  "$(sed -n 's/.*ws_prune_fraction = \([0-9.]*\).*/\1/p' "$TRAIN_LOG" | head -1)"
else
    echo "⚠  No existe $TRAIN_LOG."
    if [ -z "${GABOR_KERNEL:-}" ]; then
        echo "⚠  Y el bloque OVERRIDE está comentado => se usarían los DEFAULTS"
        echo "   (kernel=legacy, freq=norm, f1=0), que es EXACTAMENTE el bug de run5."
        echo "   Descomenta el OVERRIDE con los valores de la línea [GABOR] del train. ABORTO."
        exit 1
    fi
    echo "   Usando el bloque OVERRIDE."
fi

# La línea [GABOR] esperada, normalizada (sin timestamp y con el f1 autocalibrado
# neutralizado: en train imprime 0 y en render el valor ya resuelto).
norm_gabor() { sed 's/ \[[0-9].*$//; s/f1=[0-9.]* <- [^|]*/f1=<cfg>/'; }
EXPECTED=$(grep -am1 '^\[GABOR\] kernel=' "$TRAIN_LOG" 2>/dev/null | norm_gabor)

echo "  GABOR_KERNEL=${GABOR_KERNEL:-}  GABOR_FREQ=${GABOR_FREQ:-}  GABOR_GAMMA=${GABOR_GAMMA:-}"
echo "  GABOR_KAPPA=${GABOR_KAPPA:-}    GABOR_F1=${GABOR_F1:-}"
echo "  SCALE_CLAMP_FACTOR=${SCALE_CLAMP_FACTOR:-}  SIZE_PRUNE_MODE=${SIZE_PRUNE_MODE:-}  WS_PRUNE_FRACTION=${WS_PRUNE_FRACTION:-}"
echo

# ═══ 2) Vídeo de trayectoria (vistas nuevas interpoladas) ══════════════════
# --skip_train --skip_test --skip_mesh + --render_path => SOLO el vídeo.
# Salida: $MODEL/traj/ours_$ITER/render_traj_color.mp4
# OJO: sin --render_path esta carpeta NO se regenera. Si re-renderizas por un fallo
# de config, hay que rehacer traj Y test o te quedas con vídeos del kernel viejo
# (pasó con run5: metrics.py solo lee test/, así que el PSNR salía bien y el vídeo mal).
python render.py -s Datasets/${DATASET} \
    -m $MODEL \
    --iteration $ITER \
    --skip_train --skip_test --skip_mesh \
    --render_path \
    2>&1 | tee logs/${DATASET}${RUN}_traj.log

# ═══ 3) Comparativas render|GT de las vistas de test ═══════════════════════
# --skip_train --skip_mesh => exporta SOLO test (sin vídeo ni malla).
# Salida: $MODEL/test/ours_$ITER/vis/ (render|GT), .../renders, .../gt
# Es la carpeta que lee metrics.py.
python render.py -s Datasets/${DATASET} \
    -m $MODEL \
    --iteration $ITER \
    --skip_train --skip_mesh \
    2>&1 | tee logs/${DATASET}${RUN}_test.log

# ═══ 4) VERIFICACIÓN: el kernel del render == el del train ═════════════════
# Dos comprobaciones; las dos habrían cazado el fallo de run5 al instante:
#   (a) la línea [GABOR] del render idéntica a la del train;
#   (b) NADA de "[A] load_ply: ... PROYECTADOS" => si sale, la cota de Σaₙ del
#       render es más estrecha que la del train y el modelo se ha MUTADO al cargarlo.
echo
echo "════════════════ VERIFICACIÓN train ↔ render ════════════════"
FAIL=0
for L in logs/${DATASET}${RUN}_traj.log logs/${DATASET}${RUN}_test.log; do
    [ -f "$L" ] || continue
    GOT=$(grep -am1 '^\[GABOR\] kernel=' "$L" | norm_gabor)
    echo "  $L"
    echo "    train : ${EXPECTED:-<sin log de train>}"
    echo "    render: ${GOT:-<sin línea [GABOR]>}"
    if [ -n "$EXPECTED" ] && [ "$GOT" != "$EXPECTED" ]; then
        echo "    ❌ NO COINCIDEN — el render usó otro kernel. Métricas e imágenes NO válidas."
        FAIL=1
    fi
    if grep -qa '\[A\] load_ply.*PROYECTADOS' "$L"; then
        echo "    ❌ load_ply REPROYECTÓ coeficientes: el modelo se ha mutado en el render."
        grep -am1 '\[A\] load_ply' "$L" | sed 's/^/       /'
        FAIL=1
    fi
done
if [ -z "$EXPECTED" ]; then
    echo "  ⚠  Sin línea [GABOR] en el log del train: la comparación de kernel NO se ha"
    echo "     podido hacer. Solo se ha comprobado que load_ply no reproyectara."
fi
if [ $FAIL -eq 0 ]; then
    echo "  ✅ Sin inconsistencias detectadas. Ya se puede medir:"
    echo "     python metrics.py -m $MODEL"
else
    echo
    echo "  ⛔ NO lances metrics.py con estos renders. Revisa las env vars y repite."
    exit 1
fi
