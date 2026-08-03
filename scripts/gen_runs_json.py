#!/usr/bin/env python3
"""Genera docs/runs.json a partir de historial_runs.csv.

El CSV sigue siendo la fuente de verdad de las metricas. Este script lo pasa a un
JSON estructurado (una entrada por run, con sus parametros separados por bloques)
que es lo que lee docs/comparador_runs.html.

    python3 scripts/gen_runs_json.py

Para anadir un run: mete la fila en historial_runs.csv, anade sus parametros a
PARAMS_GABOR / TAU si es un run Gabor, y vuelve a lanzar el script.
"""

import csv
import json
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, 'historial_runs.csv')
OUT = os.path.join(RAIZ, 'docs', 'runs.json')

# ── Baselines oficiales del 2DGS ────────────────────────────────────────────
# flowers y bonsai salen del propio CSV (filas original_2dgs); kitchen esta
# documentado en CLAUDE.md. bicycle no tiene baseline medido.
ESCENAS = {
    'flowers': {
        'nombre': 'flowers',
        'tipo': '360 exterior',
        'baseline_2dgs': {'psnr': 20.89, 'ssim': 0.556, 'lpips': 0.402, 'n_splats': None},
        'tent_ab': 'run67',
        'nota': 'El atomo EMPATA con el tent. La onda no compra nada aqui: 93 % de splats apagados.',
    },
    'bonsai': {
        'nombre': 'bonsai',
        'tipo': 'interior acotado',
        'baseline_2dgs': {'psnr': 31.36, 'ssim': 0.9359, 'lpips': 0.2042, 'n_splats': 798000},
        'tent_ab': 'run76',
        'nota': 'UNICA escena donde el atomo GANA al tent en las tres, y con un 10 % menos de splats.',
    },
    'kitchen': {
        'nombre': 'kitchen',
        'tipo': 'interior acotado',
        'baseline_2dgs': {'psnr': 30.3389, 'ssim': 0.9210, 'lpips': 0.1383, 'n_splats': None},
        'tent_ab': 'run78',
        'nota': 'El atomo pierde ~0,29 dB. Refutado que sea f1 (7 puntos) y que sea capacidad (run13).',
    },
    'bicycle': {
        'nombre': 'bicycle',
        'tipo': '360 exterior',
        'baseline_2dgs': None,
        'tent_ab': None,
        'nota': 'Un solo run propio, sin baseline oficial medido ni brazo Gabor.',
    },
}

FAMILIAS = {
    'gabor_atomo':  {'etiqueta': 'Gabor atomo',   'desc': 'kernel (1-r)^beta * S(phi*t) con pedestal: a=0 recupera el tent exacto'},
    'gabor_legacy': {'etiqueta': 'Gabor legacy',  'desc': 'kernel (f/f0)^beta sin envolvente: a=0 da una CAJA. Linea abandonada'},
    'tent':         {'etiqueta': 'Clasico (tent)','desc': 'densificacion clasica, kernel tent fijo. Es el brazo de control del A/B'},
    'mcmc':         {'etiqueta': 'MCMC',          'desc': 'densificacion MCMC con cap_max fijo'},
}

# ── Parametros del kernel Gabor, verificados en los logs y en los scripts ────
# f1 va en rad por unidad de spatial_lr_scale. "auto" = autocalibrado por escena.
PARAMS_GABOR = {
    'run1_gabor':  {'kernel': 'legacy', 'freq': 'norm',  'gamma': None, 'kappa': None, 'f1': None,      'f1_origen': None,   'sum_a_max': 0.5,    'clamp_fix': False},
    'run2_gabor':  {'kernel': 'legacy', 'freq': 'norm',  'gamma': None, 'kappa': None, 'f1': None,      'f1_origen': None,   'sum_a_max': 0.5,    'clamp_fix': True},
    'run4_gabor':  {'kernel': 'legacy', 'freq': 'norm',  'gamma': None, 'kappa': None, 'f1': None,      'f1_origen': None,   'sum_a_max': 0.5,    'clamp_fix': True},
    'run5_gabor':  {'kernel': 'dir',    'freq': 'world', 'gamma': 0.3,  'kappa': 1.0,  'f1': 46.6247,   'f1_origen': 'auto', 'sum_a_max': 0.5882, 'clamp_fix': True},
    'run6_gabor':  {'kernel': 'dir',    'freq': 'world', 'gamma': 0.3,  'kappa': 1.0,  'f1': 93.2494,   'f1_origen': 'env',  'sum_a_max': 0.5882, 'clamp_fix': True},
    'run7_gabor':  {'kernel': 'dir',    'freq': 'world', 'gamma': 0.3,  'kappa': 1.0,  'f1': 186.4988,  'f1_origen': 'env',  'sum_a_max': 0.5882, 'clamp_fix': True},
    'run8_gabor':  {'kernel': 'dir',    'freq': 'world', 'gamma': 0.3,  'kappa': 1.0,  'f1': 204.3105,  'f1_origen': 'auto', 'sum_a_max': 0.5882, 'clamp_fix': True},
    'run9_gabor':  {'kernel': 'dir',    'freq': 'world', 'gamma': 0.3,  'kappa': 1.0,  'f1': 932.6041,  'f1_origen': 'auto', 'sum_a_max': 0.5882, 'clamp_fix': True},
    'run10_gabor': {'kernel': 'dir',    'freq': 'world', 'gamma': 0.3,  'kappa': 1.0,  'f1': 466.3021,  'f1_origen': 'env',  'sum_a_max': 0.5882, 'clamp_fix': True},
    'run11_gabor': {'kernel': 'dir',    'freq': 'world', 'gamma': 0.3,  'kappa': 1.0,  'f1': 233.1510,  'f1_origen': 'env',  'sum_a_max': 0.5882, 'clamp_fix': True},
    'run12_gabor': {'kernel': 'dir',    'freq': 'world', 'gamma': 0.3,  'kappa': 1.0,  'f1': 116.5755,  'f1_origen': 'env',  'sum_a_max': 0.5882, 'clamp_fix': True},
    'run13_gabor': {'kernel': 'dir',    'freq': 'world', 'gamma': 0.3,  'kappa': 1.0,  'f1': 932.6041,  'f1_origen': 'env',  'sum_a_max': 0.5882, 'clamp_fix': True},
}

# Diagnostico del kernel a 30 k, leido de la ultima linea [GABOR-W] / [A] del log.
# run6_gabor: el log local esta incompleto; los valores vienen de CLAUDE.md y del
# veredicto de run7 en el CSV. a1/a2/a3 no quedaron registrados -> null.
DIAG_GABOR = {
    'run5_gabor':  {'fw_p50': 0.082, 'fw_p90': 0.356, 'apagados': 93.40, 'encendidos': 2.91,  'suma_a': 0.2909, 'casi_planos': 21.444, 'a1': 0.1243, 'a2': 0.0791, 'a3': 0.0874},
    'run6_gabor':  {'fw_p50': 0.167, 'fw_p90': 0.746, 'apagados': 83.82, 'encendidos': 7.14,  'suma_a': 0.3103, 'casi_planos': 17.62,  'a1': None,   'a2': None,   'a3': None},
    'run7_gabor':  {'fw_p50': 0.328, 'fw_p90': 1.564, 'apagados': 64.12, 'encendidos': 17.25, 'suma_a': 0.3429, 'casi_planos': 12.912, 'a1': 0.1153, 'a2': 0.1039, 'a3': 0.1237},
    'run8_gabor':  {'fw_p50': 0.773, 'fw_p90': 3.468, 'apagados': 34.01, 'encendidos': 40.69, 'suma_a': 0.2405, 'casi_planos': 24.871, 'a1': 0.1030, 'a2': 0.0723, 'a3': 0.0652},
    'run9_gabor':  {'fw_p50': 1.600, 'fw_p90': 7.224, 'apagados': 13.51, 'encendidos': 65.54, 'suma_a': 0.2718, 'casi_planos': 18.220, 'a1': 0.1156, 'a2': 0.0829, 'a3': 0.0733},
    'run10_gabor': {'fw_p50': 0.775, 'fw_p90': 3.710, 'apagados': 35.64, 'encendidos': 42.29, 'suma_a': 0.2935, 'casi_planos': 16.398, 'a1': 0.1214, 'a2': 0.0848, 'a3': 0.0873},
    'run11_gabor': {'fw_p50': 0.374, 'fw_p90': 1.861, 'apagados': 59.30, 'encendidos': 21.13, 'suma_a': 0.2703, 'casi_planos': 19.716, 'a1': 0.0959, 'a2': 0.0902, 'a3': 0.0842},
    'run12_gabor': {'fw_p50': 0.185, 'fw_p90': 0.939, 'apagados': 79.35, 'encendidos': 9.24,  'suma_a': 0.2505, 'casi_planos': 25.319, 'a1': 0.0853, 'a2': 0.0802, 'a3': 0.0851},
    'run13_gabor': {'fw_p50': 1.455, 'fw_p90': 6.217, 'apagados': 15.62, 'encendidos': 62.42, 'suma_a': 0.2981, 'casi_planos': 14.168, 'a1': 0.1220, 'a2': 0.0930, 'a3': 0.0831},
}

# densify_grad_threshold. Verificado en el log (linea grad_pos(thr=...)) para los
# runs cuyo log esta en local; para los clasicos limpios sale de
# train_CLASSIC_server.sh, que es el script que los produjo. null = sin verificar.
TAU = {
    'run1_gabor': 2.0e-4, 'run2_gabor': 2.0e-4, 'run4_gabor': 2.0e-4,
    'run5_gabor': 2.0e-4, 'run6_gabor': 2.0e-4, 'run7_gabor': 2.0e-4,
    'run8_gabor': 2.0e-4, 'run9_gabor': 2.0e-4, 'run10_gabor': 2.0e-4,
    'run11_gabor': 2.0e-4, 'run12_gabor': 2.0e-4,
    'run13_gabor': 1.5e-4,          # <- EL delta de run13
    'run67': 2.0e-4, 'run76': 2.0e-4, 'run78': 2.0e-4,
}

# Los tres runs tent que son el brazo de control del A/B contra el atomo.
TENT_AB = {'run67', 'run76', 'run78'}

# Prune de tamano: off en todo salvo donde se probo (run3 colapso y no tiene fila).
PRUNE_TAM = {'run4_gabor': 'world'}


def num(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None


def entero(v):
    f = num(v)
    return int(f) if f is not None else None


def familia_de(run, mode):
    if run.endswith('_gabor'):
        n = int(run.replace('run', '').replace('_gabor', ''))
        return 'gabor_atomo' if n >= 5 else 'gabor_legacy'
    return 'mcmc' if mode == 'mcmc' else 'tent'


def etiqueta_de(run):
    return run.replace('_gabor', '').replace('run', 'run ') if run.startswith('run') else run


def main():
    runs = []
    with open(CSV, newline='') as f:
        for r in csv.DictReader(f):
            rid = r['run']
            if rid == 'original_2dgs':
                continue                      # va en ESCENAS como baseline
            fam = familia_de(rid, r['mode'])
            psnr, ssim, lpips = num(r['psnr']), num(r['ssim']), num(r['lpips'])
            runs.append({
                'id': rid,
                'etiqueta': etiqueta_de(rid),
                'escena': r['dataset'],
                'familia': fam,
                'tent_ab': rid in TENT_AB,
                'metricas': {
                    'psnr': psnr,
                    'ssim': ssim,
                    'lpips': lpips,
                    'fuente': r['metric_source'],
                    'honesto': r['metric_source'] == 'honest' and psnr is not None,
                    'psnr_intrain': num(r['psnr_intrain']),
                },
                'modelo': {
                    'n_splats': entero(r['N_splats']),
                    'iters': entero(r['iters']),
                    'modo': r['mode'],
                    'cap_max': entero(r['cap_max']),
                },
                'optim': {
                    'tau': TAU.get(rid),
                    'opacity_reg': num(r['opacity_reg']),
                    'scale_reg': num(r['scale_reg']),
                    'lambda_dist': num(r['lambda_dist']),
                    'dead_sustain': num(r['dead_sustain']),
                    'cov_noise_normal': num(r['cov_noise_normal']),
                    'jitter': num(r['jitter']),
                    'error_weight': num(r['error_weight']),
                    'jitter_scale': num(r['jitter_scale']),
                    'prune_tam': PRUNE_TAM.get(rid, 'off'),
                },
                'gabor': PARAMS_GABOR.get(rid),
                'diagnostico': DIAG_GABOR.get(rid),
                'cambio_clave': r['key_change'],
                'veredicto': r['verdict'],
            })

    doc = {
        'meta': {
            'proyecto': '2D Gaussian Splatting -> Beta / Gabor Splatting',
            'generado_por': 'scripts/gen_runs_json.py',
            'fuente': 'historial_runs.csv',
            'fecha': '2026-08-03',
            'nota_metricas': ('Solo cuentan las metricas HONESTAS (metrics.py sobre un modelo '
                              'entrenado con --eval). El eval in-train de train.py va por encima '
                              '(~1,4 dB en MCMC), asi que comparar contra el es un espejismo.'),
        },
        'escenas': ESCENAS,
        'familias': FAMILIAS,
        'runs': runs,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)

    honestos = sum(1 for x in runs if x['metricas']['honesto'])
    print(f'{OUT}: {len(runs)} runs ({honestos} con metricas honestas)')
    for e in ESCENAS:
        n = sum(1 for x in runs if x['escena'] == e)
        print(f'  {e:9s} {n:3d} runs')


if __name__ == '__main__':
    main()
