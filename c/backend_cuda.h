#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

/* COLI_CUDA_DLLEXPORT marks functions exported from coli_cuda.dll on Windows.
 * Define COLI_CUDA_BUILDING_DLL when compiling the .cu into the DLL (so the
 * functions are __declspec(dllexport)); the host loader does NOT include this
 * header's declarations — it resolves symbols at runtime via GetProcAddress. */
#if defined(_WIN32) && defined(COLI_CUDA_BUILDING_DLL)
#define COLI_CUDA_DLLEXPORT __declspec(dllexport)
#else
#define COLI_CUDA_DLLEXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
COLI_CUDA_DLLEXPORT int coli_cuda_init(const int *devices, int count);
COLI_CUDA_DLLEXPORT void coli_cuda_shutdown(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_count(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_at(int index);
COLI_CUDA_DLLEXPORT int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
/* device < 0 returns aggregate statistics for all configured devices. */
COLI_CUDA_DLLEXPORT void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);

/* Upload without executing, so capacity failures happen during model startup. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);

/* Upload one expert's gate/up/down tensors as a single device allocation.
 * Six separate cudaMallocs per expert waste ~2 MB-granule padding on each
 * multi-MB weight tensor (~25% of VRAM at typical int4 expert sizes); the slab
 * wastes at most one granule. gate owns the allocation, up/down borrow into it
 * (coli_cuda_tensor_bytes reports the whole slab on gate and 0 on the others,
 * so summing a triple stays correct). Free all three via coli_cuda_tensor_free
 * as usual, in any order. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_upload(
                            ColiCudaTensor **gt, ColiCudaTensor **ut, ColiCudaTensor **dt,
                            const void *gw, const float *gs, int gf, int gI, int gO,
                            const void *uw, const float *us, int uf, int uI, int uO,
                            const void *dw, const float *ds, int df, int dI, int dO,
                            int device);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2.
 * The first successful call uploads W and its row scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
COLI_CUDA_DLLEXPORT int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device);

/* Fused expert pipeline: y = down(silu(gate(x)) * up(x)).  All three tensors
 * must already be resident on one device.  Activations cross PCIe once in
 * each direction instead of once per matrix. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S);

/* Packed group of same-shaped experts. Inputs and outputs contain sum(rows)
 * consecutive [D] rows in call order. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);

/* Split form of coli_cuda_expert_group for overlapping GPU expert work with CPU
 * work: _submit enqueues on the device stream and returns without syncing; the
 * caller does CPU-side work, then _finish syncs and lands the result into the `y`
 * passed at submit. At most one submit may be outstanding per device. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_submit(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_finish(int device);

/* Same as coli_cuda_expert_group but for NON-resident int4 experts: weights are
 * streamed from host slabs into device scratch each call (PCIe-bound). All experts
 * share shape (D,I) and fmt int4. rows/x/y layout matches coli_cuda_expert_group. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_stream(
                           const void *const *gw, const void *const *uw, const void *const *dw,
                           const float *const *gs, const float *const *us, const float *const *ds,
                           const int *rows, int count, int D, int I, int device,
                           float *y, const float *x);

/* Decode-only MLA weight-absorption core for one token. kv_b is [H*(Q+V),K]. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

/* GQA decode/prefill attention: ctx[S,H,hd] from q[S,H,hd] and float K/V caches. */
COLI_CUDA_DLLEXPORT int coli_cuda_gqa_attention(float *ctx, const float *q,
                            const float *k_cache, const float *v_cache,
                            int S, int H, int Hkv, int hd, int st0, int pos_base,
                            int max_t, int device);

/* Resident-cache GQA attention: appends the S new K/V rows (k_new/v_new,
 * [S,Hkv,hd]) into a per-layer fp16 device cache and runs attention over the
 * resident cache, so only the new rows, q and ctx cross PCIe each token. */
COLI_CUDA_DLLEXPORT int coli_cuda_gqa_attention_cached(int layer, float *ctx, const float *q,
                            const float *k_new, const float *v_new,
                            int S, int H, int Hkv, int hd, int st0, int pos_base,
                            int max_t, int device);

COLI_CUDA_DLLEXPORT void coli_cuda_tensor_free(ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
