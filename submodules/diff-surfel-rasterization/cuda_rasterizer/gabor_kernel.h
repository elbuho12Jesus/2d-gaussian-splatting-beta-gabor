/*
 * Kernel Gabor como ÁTOMO: envolvente × onda.
 *
 * Este header es la ÚNICA definición del kernel y de sus derivadas. forward.cu y
 * backward.cu lo incluyen los dos a propósito: en este proyecto ya se pagó una vez el
 * precio de tener la fórmula duplicada (el clamp de alpha del backward que no coincidía
 * con el del forward, run60/62), así que aquí se define una sola vez.
 *
 * ---------------------------------------------------------------------------
 * LAYOUT del tensor por-Gaussiana `a` (GABOR_STRIDE = 5 floats por splat)
 *   [0..2]  a1,a2,a3  coeficientes de la serie de cosenos (a_n >= 0, sum <= limite)
 *   [3]     phi       frecuencia base efectiva, ADIMENSIONAL
 *   [4]     b         pedestal
 *
 * phi y b se calculan en PYTHON (GaussianModel.get_gabor) de forma diferenciable, y
 * CUDA los trata como parámetros independientes: devuelve dL/dphi y dL/db y es autograd
 * quien compone la cadena hacia _a y _scaling. Por eso aquí no aparecen ni gamma ni f1.
 *
 *   modo world:  phi = f1 * s_u   (s_u = escala del eje u en unidades de mundo)
 *   modo norm :  phi = pi          (medio periodo dentro del soporte = geometría vieja)
 *   pedestal  :  b   = gamma + (1-gamma) * (1 - sum(a_n))
 *
 * ---------------------------------------------------------------------------
 * MODOS
 *   LEGACY  kernel = (f/f0)^beta,  f = 1/2 + sum a_n cos((2n-1) pi r)
 *           El kernel de run1..run4. Ignora phi y b. Su punto de degeneración (a=0) es
 *           una CAJA, y no separa envolvente de onda: ver docs/rediseno_kernel_gabor_adagar.html
 *   RADIAL  kernel = (1-r)^beta * S(phi*r)      <- cambio 1 (pedestal), modulación radial
 *   DIR     kernel = (1-r)^beta * S(phi*u)      <- cambio 1 + cambio 3 (direccional)
 *
 *   En RADIAL/DIR, a = 0 => b = 1 => S == 1 => kernel = (1-r)^beta = el tent de run67
 *   EXACTO. El baseline es el punto neutro del espacio de parámetros, no una esquina.
 *
 * ---------------------------------------------------------------------------
 * INTERRUPTOR DE ESCALA (docs §3.1-§3.2)
 *   La modulación solo se aplica donde manda la geometría real (rho3d <= rho2d). Si el
 *   splat es sub-píxel y quien manda es el filtro paso-bajo de pantalla, S se fuerza a 1
 *   (Gaussiana/tent pura): un splat sub-píxel no puede mostrar textura, y así el gradiente
 *   de la rama rho2d se queda como estaba (solo geometría), sin términos cruzados.
 */

#ifndef CUDA_RASTERIZER_GABOR_H_INCLUDED
#define CUDA_RASTERIZER_GABOR_H_INCLUDED

#define GABOR_STRIDE 5

#define GABOR_MODE_LEGACY 0
#define GABOR_MODE_RADIAL 1
#define GABOR_MODE_DIR    2

#define GABOR_PI 3.14159265358979323846f

// Onda S(t) = b + sum_n a_n cos(w_n t),  w_n = (2n-1) * phi.
// Devuelve S y, por referencia, dS/dt y dS/dphi (esta última la necesita el backward).
__device__ inline float gabor_wave(
	float a1, float a2, float a3, float b, float phi, float t,
	float& dS_dt, float& dS_dphi)
{
	const float w1 = phi, w2 = 3.0f * phi, w3 = 5.0f * phi;
	const float s1 = sinf(w1 * t), s2 = sinf(w2 * t), s3 = sinf(w3 * t);
	dS_dt   = -(a1 * w1 * s1 + a2 * w2 * s2 + a3 * w3 * s3);
	// dS/dphi = sum_n a_n * (-(2n-1) * t * sin(w_n t))
	dS_dphi = -t * (a1 * 1.0f * s1 + a2 * 3.0f * s2 + a3 * 5.0f * s3);
	return b + a1 * cosf(w1 * t) + a2 * cosf(w2 * t) + a3 * cosf(w3 * t);
}

// Envolvente de soporte compacto E(r) = (1-r)^beta, y su derivada dE/dr.
// El guard 1e-6 sobre (1-r) evita pow(0, beta<1) = inf en el borde; el llamador ya
// descarta r >= 1.
__device__ inline float gabor_envelope(float r, float beta, float& dE_dr)
{
	const float u = fmaxf(1.0f - r, 1e-6f);
	const float E = powf(u, beta);
	dE_dr = -beta * powf(u, beta - 1.0f);
	return E;
}

#endif
