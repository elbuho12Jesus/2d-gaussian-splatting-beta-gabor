// Comprueba docs/comparador_runs.html contra docs/runs.json SIN navegador.
//
//     node scripts/test_comparador.js
//
// Ejecuta el <script> real de la pagina sobre un DOM minimo y verifica dos cosas:
//   1. que no revienta en ninguna escena, filtro, orden de columna ni seleccion
//   2. que ninguna etiqueta de los SVG se pisa con otra ni se sale del viewBox
// Lo segundo es el sustituto de "abrirlo y mirarlo": ya ha cazado dos colisiones
// reales (el titulo contra su aclaracion, y el 2DGS contra el tent en kitchen,
// donde las dos referencias se llevan 0,05 dB).
//
// Hay que volver a lanzarlo despues de tocar el HTML o de regenerar el JSON.

const fs = require('fs');
const path = require('path');
const RAIZ = path.dirname(__dirname);
const html = fs.readFileSync(path.join(RAIZ, 'docs/comparador_runs.html'), 'utf8');
const datos = JSON.parse(fs.readFileSync(path.join(RAIZ, 'docs/runs.json'), 'utf8'));
const codigo = html.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/\ncargar\(\);\s*$/, '\n');

// alto del viewBox de cada grafico, leido del propio fuente para no desincronizarse
const ALTO_STRIP = +html.match(/const W = 1000, H = (\d+), ML/)[1];
const ALTO_ONDA = +html.match(/const W = 1000, H = (\d+), ML = 62/)[1];

/* ── DOM minimo ─────────────────────────────────────────────────────────── */
function nodo(tag) {
  return {
    tagName: tag, children: [], attrs: {}, listeners: {}, _text: '', _html: '',
    className: '', hidden: false,
    set textContent(v) { this._text = String(v); this.children = []; },
    get textContent() { return this._text; },
    // asignar innerHTML reemplaza los hijos, igual que en el DOM real: si no,
    // `cont.innerHTML=''` no limpia nada y el arbol crece sin fin
    set innerHTML(v) { this._html = String(v); this.children = []; },
    get innerHTML() { return this._html || ''; },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(t, f) { (this.listeners[t] ||= []).push(f); },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 1000, height: 300 }),
    style: new Proxy({}, { set: () => true, get: () => '' }),
  };
}
const IDS = ['loader', 'app', 'scenes', 'filters', 'strips', 'legend', 'thead', 'tbody',
             'tblhint', 'detail', 'wave', 'wavelegend', 'wavenote', 'themebtn', 'tbl'];
const porId = {};
for (const i of IDS) porId[i] = nodo('div');
global.document = {
  createElement: nodo,
  createElementNS: (ns, t) => nodo(t),
  createTextNode: t => ({ nodeType: 3, textContent: String(t) }),
  getElementById: id => porId[id] || nodo('div'),
  documentElement: nodo('html'),
};
global.matchMedia = () => ({ matches: false });

const api = new Function(codigo + '\nreturn {arrancar, ESTADO, redibujar, pintarOnda, FAM, tentDe, base2dgs};')();

/* ── 1 · que no reviente ────────────────────────────────────────────────── */
let fallos = 0;
api.arrancar(datos);
console.log('✓ arranca sin excepcion');

const escenas = [...new Set(datos.runs.filter(r => r.metricas.honesto).map(r => r.escena))];
for (const esc of escenas) {
  api.ESTADO.escena = esc; api.ESTADO.sel = null;
  api.ESTADO.familias = new Set(Object.keys(api.FAM));
  api.redibujar();
  const t = api.tentDe(esc), o = api.base2dgs(esc);
  console.log(`✓ ${esc.padEnd(8)} ${String(porId.tbody.children.length).padStart(3)} filas · ` +
              `tent=${t ? t.id : '—'} · 2DGS=${o ? o.psnr : '—'}`);
  for (const f of Object.keys(api.FAM)) {           // cada familia por separado
    api.ESTADO.familias = new Set([f]); api.redibujar();
  }
  api.ESTADO.familias = new Set(Object.keys(api.FAM)); api.redibujar();
  for (const th of porId.thead.children) {          // cada columna, en los dos sentidos
    const click = th.listeners.click?.[0];
    if (click) { click(); click(); }
  }
  for (const r of datos.runs.filter(x => x.escena === esc && x.metricas.honesto)) {
    api.ESTADO.sel = r.id; api.redibujar();         // panel de detalle de cada run
  }
  api.ESTADO.sel = null;
}
console.log('✓ todas las escenas, familias, ordenes y selecciones sin excepcion');

/* ── 2 · etiquetas que se pisan o se salen ──────────────────────────────── */
const texto = n => (n.children || []).map(c => c.nodeType === 3 ? c.textContent : texto(c)).join('') || n._text || '';
const svgs = (n, out = []) => { if (n.tagName === 'svg') out.push(n); for (const c of n.children || []) if (c.children) svgs(c, out); return out; };
const textos = (n, out = []) => { if (n.tagName === 'text') out.push(n); for (const c of n.children || []) if (c.children) textos(c, out); return out; };

// caja aproximada de un <text> SVG: ancho ~ 0,55 * tamano por caracter
function caja(t) {
  const s = +(t.attrs['font-size'] || 12), x = +t.attrs.x, y = +t.attrs.y;
  const txt = texto(t), w = txt.length * s * 0.55, a = t.attrs['text-anchor'] || 'start';
  const x0 = a === 'middle' ? x - w / 2 : a === 'end' ? x - w : x;
  return { x0, x1: x0 + w, y0: y - s * 0.8, y1: y + s * 0.25, txt };
}
const pisan = (a, b) => a.x0 < b.x1 - 1 && b.x0 < a.x1 - 1 && a.y0 < b.y1 - 1 && b.y0 < a.y1 - 1;

let etiquetas = 0;
function revisar(donde, contenedor, alto) {
  for (const sv of svgs(contenedor)) {
    const cajas = textos(sv).map(caja);
    etiquetas += cajas.length;
    for (let i = 0; i < cajas.length; i++) {
      const c = cajas[i];
      if (c.x0 < -2 || c.x1 > 1002) { console.log(`  ✗ [${donde}] se sale a lo ancho: "${c.txt}"`); fallos++; }
      if (c.y0 < -2 || c.y1 > alto + 2) { console.log(`  ✗ [${donde}] se sale a lo alto: "${c.txt}"`); fallos++; }
      for (let j = i + 1; j < cajas.length; j++) {
        if (pisan(c, cajas[j])) { console.log(`  ✗ [${donde}] se pisan: "${c.txt}" ↔ "${cajas[j].txt}"`); fallos++; }
      }
    }
  }
}
for (const esc of escenas) {
  api.ESTADO.escena = esc; api.ESTADO.sel = null;
  api.ESTADO.familias = new Set(Object.keys(api.FAM)); api.redibujar();
  revisar(esc, porId.strips, ALTO_STRIP);
  for (const f of Object.keys(api.FAM)) {   // filtrar mueve el mejor y su etiqueta directa
    api.ESTADO.familias = new Set([f]); api.redibujar();
    revisar(`${esc}/${f}`, porId.strips, ALTO_STRIP);
  }
}
api.pintarOnda();
revisar('onda', porId.wave, ALTO_ONDA);

console.log(`${fallos ? '✗' : '✓'} ${etiquetas} etiquetas revisadas · ${fallos} problema(s)`);
process.exit(fallos ? 1 : 0);
