#include "backend_cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
    /* Slab-upload members (coli_cuda_expert_upload): the three tensors of one
     * expert share a single cudaMalloc. `slab` is set on the owning tensor only
     * (its weights are section 0, but the whole allocation is freed via slab);
     * `borrowed` tensors point into the owner's slab and free nothing on the
     * device. Per-tensor uploads leave both fields zero. */
    void *slab; size_t slab_bytes;
    int borrowed;
};

typedef struct {
    int device;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y; size_t host_x_cap,host_y_cap;
    /* Dense-matmul scratch, kept separate from the expert-group buffers above so a
     * dense call (QKV/O/shared/lm_head under CUDA_DENSE) never aliases an in-flight
     * grouped expert submit sharing the stream. */
    float *mm_x,*mm_y; size_t mm_x_cap,mm_y_cap;
    float *mm_hx,*mm_hy; size_t mm_hx_cap,mm_hy_cap;
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    float *gq,*gk,*gv,*gctx,*gsc; size_t gq_cap,gk_cap,gv_cap,gctx_cap,gsc_cap;
    cudaStream_t stream;
    void *group_desc; size_t group_desc_cap;
    /* JIT-streaming scratch: non-resident expert weights uploaded per call. */
    void *sw_g,*sw_u,*sw_d; size_t sw_g_cap,sw_u_cap,sw_d_cap;
    float *ss_g,*ss_u,*ss_d; size_t ss_g_cap,ss_u_cap,ss_d_cap;
    /* Async grouped-expert pipeline state (submit/finish overlap): a grouped call
     * enqueued on `stream` but not yet synced. Exactly one may be outstanding per
     * device — the caller submits, does CPU-side work, then finishes. */
    int group_pending;
    float *pend_y; size_t pend_bytes;                 /* caller dest + size for the finish memcpy */
    uint64_t pend_experts, pend_rows;
    int pend_profile; cudaEvent_t pend_ev[4];
    size_t tensor_count, tensor_bytes;
} DeviceContext;

/* Resident fp16 KV cache for GPU attention (CUDA_ATTN): allocated once per layer
 * and appended to per token, so decode uploads only the new row(s) instead of
 * re-sending the whole cache every call. Indexed by absolute layer number. */
typedef struct { __half *k, *v; int max_t, hd, Hkv, device; } KVCache;
#define COLI_CUDA_MAX_LAYERS 256
static KVCache g_kv[COLI_CUDA_MAX_LAYERS];
static void kv_cache_free_all(void);   /* defined below; used by coli_cuda_shutdown */

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static std::mutex g_group_stats_mu;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

static int select_ctx(DeviceContext *ctx) {
    return ctx && cuda_ok(cudaSetDevice(ctx->device), "select device");
}

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2) {
        uint8_t v = q[i >> 1];
        int n=(i&1)?(v>>4):(v&15); return static_cast<float>(n&8?n-16:n);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

__global__ static void offset_to_signed_s4(uint8_t *q,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]^=0x88;
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        sum += xs[i] * weight_at(weights, fmt, row, i);

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = partial[0] * (fmt ? scales[o] : 1.0f);
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

__global__ static void quantize_s4_rows(uint8_t *q,float *scale,const float *x,int S,int K){
    int s=blockIdx.x; if(s>=S)return; const float *xs=x+(size_t)s*K;
    float v=0; for(int i=threadIdx.x;i<K;i+=blockDim.x)v=fmaxf(v,fabsf(xs[i]));
    __shared__ float m[256]; m[threadIdx.x]=v; __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)m[threadIdx.x]=fmaxf(m[threadIdx.x],m[threadIdx.x+n]);__syncthreads();}
    float sc=m[0]>0?m[0]/7.f:1.f; if(!threadIdx.x)scale[s]=sc;
    uint8_t *dst=q+(size_t)s*((K+1)/2);
    for(int b=threadIdx.x;b<(K+1)/2;b+=blockDim.x){
        int i=b*2,a=__float2int_rn(xs[i]/sc),c=i+1<K?__float2int_rn(xs[i+1]/sc):0;
        a=max(-8,min(7,a)); c=max(-8,min(7,c)); dst[b]=(uint8_t)((a&15)|((c&15)<<4));
    }
}

__global__ static void grouped_s4_wmma(float *y,const uint8_t *x,const float *xscale,
                                        const GroupDesc *desc,int K,int O,int which){
/* Sub-byte (s4) WMMA exists only on sm_75..sm_89 (Turing/Ampere/Ada). Hopper (sm_90)
 * deprecated it and Blackwell (sm_120) + CUDA 12.8 removed the type entirely, so
 * referencing it there fails to compile. Compile the body only where it's valid;
 * this path is opt-in (COLI_CUDA_TC_INT4) and unused on Blackwell regardless. */
#if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 900
    using namespace nvcuda;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,tile=blockIdx.x*8+warp,c=blockIdx.y;
    if(tile*8>=O)return; GroupDesc d=desc[c];
    const void *w=which==0?d.g:(which==1?d.u:d.d);
    const float *ws=which==0?d.gs:(which==1?d.us:d.ds);
    int fmt=which==0?d.gf:(which==1?d.uf:d.df);
    if(fmt!=2)return;
    wmma::fragment<wmma::accumulator,8,8,32,int> acc; wmma::fill_fragment(acc,0);
    const uint8_t *a=x+(size_t)d.offset*((K+1)/2);
    const uint8_t *b=(const uint8_t*)w+(size_t)(tile*8)*((K+1)/2);
    for(int k=0;k<K;k+=32){
        wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+k/2,K);
        wmma::load_matrix_sync(bf,b+k/2,K);
        wmma::mma_sync(acc,af,bf,acc);
    }
    __shared__ int out[8][64]; wmma::store_matrix_sync(out[warp],acc,8,wmma::mem_row_major);
    for(int i=lane;i<64;i+=32){int s=i/8,o=tile*8+i%8;
        if(s<d.rows&&o<O)y[(size_t)(d.offset+s)*O+o]=(float)out[warp][i]*xscale[d.offset+s]*ws[o];}
#endif
}

__global__ static void grouped_hidden(float *y,const float *x,const GroupDesc *desc,
                                      int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0; for(int i=threadIdx.x;i<D;i+=blockDim.x) sum+=xs[i]*weight_at(w,fmt,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*I+o]=p[0]*(fmt?sc[o]:1.f);
}

__global__ static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum=0; for(int i=threadIdx.x;i<I;i+=blockDim.x) sum+=xs[i]*weight_at(d.d,d.df,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*D+o]=p[0]*(d.df?d.ds[o]:1.f);
}

__device__ static void unpack_s4(uint8_t v,float *lo,float *hi){
    int a=v&15,b=v>>4; *lo=(float)(a&8?a-16:a); *hi=(float)(b&8?b-16:b);
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
__global__ static void grouped_hidden_w4(float *y,const float *x,const GroupDesc *desc,
                                         int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0;for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*I+o]=p[0]*sc[o];
}

__global__ static void grouped_hidden_w4_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;gate[z]=gp[0]*d.gs[o];up[z]=upv[0]*d.us[o];}
}

__global__ static void grouped_down_w4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0]*d.ds[o];
}

/* ---- COLI_CUDA_FAST_GEMV kernels (default): one warp per output row instead of
 * one 256-thread block, a __shfl warp reduction instead of the shared-memory
 * tree (no __syncthreads), and byte-strided loads that coalesce across the warp.
 * grouped_hidden_w4_dual_fused also folds SiLU(gate)*up into the same pass, so the
 * separate silu_mul launch and its full extra read+write of the hidden buffer are
 * gone. Same products as the block-per-output path above, but summed in warp-stride
 * order, so results match it only to FP round-off (like any GPU-vs-CPU reduction
 * here); set COLI_CUDA_FAST_GEMV=0 for the block-per-output path. */
#define COLI_WPB 8   /* warps per 256-thread block */
__device__ __forceinline__ float warp_reduce_sum(float v){
    for(int off=16;off;off>>=1) v+=__shfl_down_sync(0xffffffffu,v,off); return v;
}
__global__ static void grouped_hidden_w4_dual_fused(float *hp,const float *x,
                                                    const GroupDesc *desc,int I,int D){
    int lane=threadIdx.x&31, o=blockIdx.x*COLI_WPB+(threadIdx.x>>5);
    if(o>=I) return;
    int s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c]; if(s>=d.rows) return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D; int nb=(D+1)/2;
    float ga=0,ua=0;
    for(int b=lane;b<nb;b+=32){
        float g0,g1,u0,u1; unpack_s4(gr[b],&g0,&g1); unpack_s4(ur[b],&u0,&u1);
        int i=b*2; float x0=xs[i]; ga+=x0*g0; ua+=x0*u0;
        if(i+1<D){ float x1=xs[i+1]; ga+=x1*g1; ua+=x1*u1; }
    }
    ga=warp_reduce_sum(ga); ua=warp_reduce_sum(ua);
    if(!lane){ float g=ga*d.gs[o], u=ua*d.us[o];
        hp[(size_t)(d.offset+s)*I+o]=(g/(1.f+expf(-g)))*u; }
}
__global__ static void grouped_down_w4_fast(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int lane=threadIdx.x&31, o=blockIdx.x*COLI_WPB+(threadIdx.x>>5);
    if(o>=D) return;
    int s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c]; if(s>=d.rows) return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I; int nb=(I+1)/2; float sum=0;
    for(int b=lane;b<nb;b+=32){ float a,z; unpack_s4(row[b],&a,&z);
        int i=b*2; sum+=xs[i]*a; if(i+1<I) sum+=xs[i+1]*z; }
    sum=warp_reduce_sum(sum);
    if(!lane) y[(size_t)(d.offset+s)*D+o]=sum*d.ds[o];
}

__global__ static void attention_absorb_kernel(float *ctx,const float *q,const float *latent,
                                                const float *rope,const void *weights,const float *wscale,
                                                int fmt,int H,int Q,int R,int V,int K,int T,float scale){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*(fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    if(!tid){float mx=scores[0];for(int t=1;t<T;t++)mx=fmaxf(mx,scores[t]);float z=0;
        for(int t=0;t<T;t++){scores[t]=expf(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);ctx[(size_t)h*V+v]=a*(fmt?wscale[row]:1.f);}
}

#define GQA_MAX_NT 8192

__global__ static void gqa_attn_kernel(float *ctx,const float *q,const float *k_cache,const float *v_cache,
                                       float *scores,int S,int H,int Hkv,int hd,int st0,int pos_base,
                                       int max_t,float scale,int nrep){
    int h=blockIdx.x,s=blockIdx.y;
    if(s>=S)return;
    int kvh=h/nrep,pos=pos_base+s,nt=pos+1-st0;
    if(nt<1||nt>GQA_MAX_NT)return;
    float *sc=scores+((size_t)s*H+h)*GQA_MAX_NT;
    const float *qv=q+((size_t)s*H+h)*hd;
    for(int jj=threadIdx.x;jj<nt;jj+=blockDim.x){
        int t=st0+jj; const float *kv=k_cache+((size_t)kvh*max_t+t)*hd;
        float dot=0; for(int d=0;d<hd;d++) dot+=qv[d]*kv[d];
        sc[jj]=dot*scale;
    }
    __syncthreads();
    if(!threadIdx.x){
        float mx=sc[0]; for(int i=1;i<nt;i++) mx=fmaxf(mx,sc[i]);
        float sum=0; for(int i=0;i<nt;i++){ sc[i]=expf(sc[i]-mx); sum+=sc[i]; }
        float inv=sum>0?1.f/sum:0.f; for(int i=0;i<nt;i++) sc[i]*=inv;
    }
    __syncthreads();
    float *cx=ctx+((size_t)s*H+h)*hd;
    for(int d=threadIdx.x;d<hd;d+=blockDim.x){
        float acc=0;
        for(int jj=0;jj<nt;jj++){
            int t=st0+jj; const float *vv=v_cache+((size_t)kvh*max_t+t)*hd;
            acc+=sc[jj]*vv[d];
        }
        cx[d]=acc;
    }
}

/* Append S new rows (one per kv-head) into the resident fp16 KV cache, converting
 * float->half. k_new/v_new are [S,Hkv,hd] (host layout, uploaded to scratch first);
 * the cache is [Hkv,max_t,hd] so row t=pos_base+s of head kvh lands contiguously. */
__global__ static void kv_append_kernel(__half *k_cache,__half *v_cache,
                                        const float *k_new,const float *v_new,
                                        int S,int Hkv,int hd,int pos_base,int max_t){
    int s=blockIdx.x,kvh=blockIdx.y; int pos=pos_base+s; if(pos>=max_t) return;
    const float *ks=k_new+((size_t)s*Hkv+kvh)*hd, *vs=v_new+((size_t)s*Hkv+kvh)*hd;
    __half *kd=k_cache+((size_t)kvh*max_t+pos)*hd, *vd=v_cache+((size_t)kvh*max_t+pos)*hd;
    for(int d=threadIdx.x;d<hd;d+=blockDim.x){ kd[d]=__float2half(ks[d]); vd[d]=__float2half(vs[d]); }
}

/* Resident-cache GQA attention: reads the fp16 K/V cache in place (no per-token
 * re-upload) and parallelizes the softmax across the block instead of the serial
 * thread-0 loop the float kernel used. One block per (s,h); scores live in the
 * caller-sized `scores` scratch (stride = nt_stride, the batch's max nt). */
__global__ static void gqa_attn_half_kernel(float *ctx,const float *q,
                                            const __half *k_cache,const __half *v_cache,
                                            float *scores,int S,int H,int Hkv,int hd,int st0,
                                            int pos_base,int max_t,int nt_stride,float scale,int nrep){
    int h=blockIdx.x,s=blockIdx.y;
    if(s>=S)return;
    int kvh=h/nrep,pos=pos_base+s,nt=pos+1-st0;
    if(nt<1||nt>nt_stride)return;
    float *sc=scores+((size_t)s*H+h)*nt_stride;
    const float *qv=q+((size_t)s*H+h)*hd;
    for(int jj=threadIdx.x;jj<nt;jj+=blockDim.x){
        int t=st0+jj; const __half *kv=k_cache+((size_t)kvh*max_t+t)*hd;
        float dot=0; for(int d=0;d<hd;d++) dot+=qv[d]*__half2float(kv[d]);
        sc[jj]=dot*scale;
    }
    __syncthreads();
    __shared__ float red[256];
    int tid=threadIdx.x;
    /* parallel max */
    float lm=-1e30f; for(int jj=tid;jj<nt;jj+=blockDim.x) lm=fmaxf(lm,sc[jj]);
    red[tid]=lm; __syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){ if(tid<n) red[tid]=fmaxf(red[tid],red[tid+n]); __syncthreads(); }
    float mx=red[0]; __syncthreads();
    /* parallel exp + sum */
    float ls=0; for(int jj=tid;jj<nt;jj+=blockDim.x){ float e=expf(sc[jj]-mx); sc[jj]=e; ls+=e; }
    red[tid]=ls; __syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){ if(tid<n) red[tid]+=red[tid+n]; __syncthreads(); }
    float inv=red[0]>0?1.f/red[0]:0.f; __syncthreads();
    float *cx=ctx+((size_t)s*H+h)*hd;
    for(int d=tid;d<hd;d+=blockDim.x){
        float acc=0;
        for(int jj=0;jj<nt;jj++){
            int t=st0+jj; const __half *vv=v_cache+((size_t)kvh*max_t+t)*hd;
            acc+=sc[jj]*__half2float(vv[d]);
        }
        cx[d]=acc*inv;
    }
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1; if(*ptr) cudaFree(*ptr); *ptr=nullptr; *cap=0;
    if(!cuda_ok(cudaMalloc(ptr,bytes),"descriptor allocation")) return 0; *cap=bytes; return 1;
}

static int reserve_pinned(float **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        if(!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking),"stream creation")){
            g_nctx=0;return 0;
        }
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_cuda_shutdown(void) {
    kv_cache_free_all();
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->group_pending && ctx->pend_profile) for(int e=0;e<4;e++) cudaEventDestroy(ctx->pend_ev[e]);
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->gate) cudaFree(ctx->gate);
        if (ctx->up) cudaFree(ctx->up);
        if (ctx->qx) cudaFree(ctx->qx);
        if (ctx->qscale) cudaFree(ctx->qscale);
        if(ctx->aq)cudaFree(ctx->aq);if(ctx->al)cudaFree(ctx->al);if(ctx->ar)cudaFree(ctx->ar);if(ctx->ac)cudaFree(ctx->ac);
        if(ctx->gq)cudaFree(ctx->gq);if(ctx->gk)cudaFree(ctx->gk);if(ctx->gv)cudaFree(ctx->gv);
        if(ctx->gctx)cudaFree(ctx->gctx);if(ctx->gsc)cudaFree(ctx->gsc);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->mm_x) cudaFree(ctx->mm_x); if (ctx->mm_y) cudaFree(ctx->mm_y);
        if (ctx->mm_hx) cudaFreeHost(ctx->mm_hx); if (ctx->mm_hy) cudaFreeHost(ctx->mm_hy);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
        if (ctx->sw_g) cudaFree(ctx->sw_g); if (ctx->sw_u) cudaFree(ctx->sw_u); if (ctx->sw_d) cudaFree(ctx->sw_d);
        if (ctx->ss_g) cudaFree(ctx->ss_g); if (ctx->ss_u) cudaFree(ctx->ss_u); if (ctx->ss_d) cudaFree(ctx->ss_d);
        ctx->sw_g=ctx->sw_u=ctx->sw_d=nullptr; ctx->ss_g=ctx->ss_u=ctx->ss_d=nullptr;
        ctx->sw_g_cap=ctx->sw_u_cap=ctx->sw_d_cap=0; ctx->ss_g_cap=ctx->ss_u_cap=ctx->ss_d_cap=0;
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->gq=ctx->gk=ctx->gv=ctx->gctx=ctx->gsc=nullptr;
        ctx->gq_cap=ctx->gk_cap=ctx->gv_cap=ctx->gctx_cap=ctx->gsc_cap=0;
        ctx->host_x=ctx->host_y=nullptr;ctx->stream=nullptr;
        ctx->mm_x=ctx->mm_y=ctx->mm_hx=ctx->mm_hy=nullptr;
        ctx->mm_x_cap=ctx->mm_y_cap=ctx->mm_hx_cap=ctx->mm_hy_cap=0;
        ctx->group_pending=0; ctx->pend_profile=0;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->host_x_cap=ctx->host_y_cap=0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
    }
    g_nctx = 0;
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if(fmt==2){offset_to_signed_s4<<<(unsigned)((t->weight_bytes+255)/256),256>>>((uint8_t*)t->weights,t->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight conversion")){coli_cuda_tensor_free(t);return 0;}}
    if (fmt) {
        if (!cuda_ok(cudaMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, (size_t)O * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

/* Upload one expert's gate/up/down weights (+ scales) as a SINGLE device
 * allocation. cudaMalloc hands out driver granules (~2 MB): six allocations per
 * expert (3 weights of a few MB + 3 tiny scale arrays) can burn ~25% more
 * physical VRAM than the logical bytes, which is why byte accounting used to
 * overshoot the card. One slab wastes at most one granule per expert. The gate
 * tensor owns the slab; up/down borrow sections and free nothing on the device.
 * Idempotent like coli_cuda_tensor_upload: if handles already exist, succeed
 * only when every shape matches. */
extern "C" int coli_cuda_expert_upload(
        ColiCudaTensor **gt, ColiCudaTensor **ut, ColiCudaTensor **dt,
        const void *gw, const float *gs, int gf, int gI, int gO,
        const void *uw, const float *us, int uf, int uI, int uO,
        const void *dw, const float *ds, int df, int dI, int dO,
        int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!gt || !ut || !dt || !gw || !uw || !dw || !select_ctx(ctx)) return 0;
    if (*gt || *ut || *dt) {
        return *gt && *ut && *dt &&
            (*gt)->fmt == gf && (*gt)->I == gI && (*gt)->O == gO && (*gt)->device == device &&
            (*ut)->fmt == uf && (*ut)->I == uI && (*ut)->O == uO && (*ut)->device == device &&
            (*dt)->fmt == df && (*dt)->I == dI && (*dt)->O == dO && (*dt)->device == device;
    }
    const void *w[3] = {gw, uw, dw};
    const float *s[3] = {gs, us, ds};
    int fm[3] = {gf, uf, df}, Is[3] = {gI, uI, dI}, Os[3] = {gO, uO, dO};
    size_t wb[3], sb[3], off[6], cur = 0;
    const size_t A = 256;   /* section alignment: covers float scales and coalesced loads */
    for (int k = 0; k < 3; k++) {
        size_t rb = row_bytes(fm[k], Is[k]);
        if (!rb || Is[k] < 1 || Os[k] < 1 || (fm[k] && !s[k])) return 0;
        wb[k] = rb * (size_t)Os[k];
        sb[k] = fm[k] ? (size_t)Os[k] * sizeof(float) : 0;
    }
    for (int k = 0; k < 3; k++) { off[k] = cur; cur += (wb[k] + A - 1) & ~(A - 1); }
    for (int k = 0; k < 3; k++) { off[3 + k] = cur; cur += (sb[k] + A - 1) & ~(A - 1); }
    uint8_t *base = nullptr;
    if (!cuda_ok(cudaMalloc(&base, cur), "expert slab allocation")) return 0;
    for (int k = 0; k < 3; k++) {
        if (!cuda_ok(cudaMemcpy(base + off[k], w[k], wb[k], cudaMemcpyHostToDevice), "expert slab upload") ||
            (sb[k] && !cuda_ok(cudaMemcpy(base + off[3 + k], s[k], sb[k], cudaMemcpyHostToDevice), "expert slab scale upload"))) {
            cudaFree(base);
            return 0;
        }
    }
    int converted = 0;
    for (int k = 0; k < 3; k++) if (fm[k] == 2) {
        offset_to_signed_s4<<<(unsigned)((wb[k] + 255) / 256), 256>>>(base + off[k], wb[k]);
        if (!cuda_ok(cudaGetLastError(), "int4 weight conversion")) { cudaFree(base); return 0; }
        converted = 1;
    }
    /* The conversion ran on the legacy stream; compute runs on the per-device
     * non-blocking stream, so make the weights visible before returning. */
    if (converted && !cuda_ok(cudaDeviceSynchronize(), "expert slab conversion sync")) { cudaFree(base); return 0; }
    ColiCudaTensor *t[3];
    for (int k = 0; k < 3; k++) {
        t[k] = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t[k])));
        if (!t[k]) { for (int z = 0; z < k; z++) std::free(t[z]); cudaFree(base); return 0; }
    }
    for (int k = 0; k < 3; k++) {
        t[k]->fmt = fm[k]; t[k]->I = Is[k]; t[k]->O = Os[k]; t[k]->device = device;
        t[k]->weights = base + off[k];
        t[k]->scales = sb[k] ? reinterpret_cast<float *>(base + off[3 + k]) : nullptr;
        t[k]->weight_bytes = wb[k];
        t[k]->tracked = 1;
        t[k]->borrowed = (k != 0);
    }
    t[0]->slab = base; t[0]->slab_bytes = cur;
    ctx->tensor_count += 3;
    ctx->tensor_bytes += cur;
    *gt = t[0]; *ut = t[1]; *dt = t[2];
    return 1;
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    /* Async on the device stream with pinned staging instead of blocking pageable
     * copies on the legacy default stream — the default stream's device-wide sync
     * would serialize against any in-flight grouped-expert work; this does not. */
    if (!reserve(&ctx->mm_x, &ctx->mm_x_cap, xb) || !reserve(&ctx->mm_y, &ctx->mm_y_cap, yb) ||
        !reserve_pinned(&ctx->mm_hx, &ctx->mm_hx_cap, xb) ||
        !reserve_pinned(&ctx->mm_hy, &ctx->mm_hy_cap, yb)) return 0;
    std::memcpy(ctx->mm_hx, x, xb);
    if (!cuda_ok(cudaMemcpyAsync(ctx->mm_x, ctx->mm_hx, xb, cudaMemcpyHostToDevice, ctx->stream), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256, 0, ctx->stream>>>(ctx->mm_y, ctx->mm_x, t->weights, t->scales, fmt, S, I, O, rb);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpyAsync(ctx->mm_hy, ctx->mm_y, yb, cudaMemcpyDeviceToHost, ctx->stream), "output download") ||
        !cuda_ok(cudaStreamSynchronize(ctx->stream), "matmul synchronize")) return 0;
    std::memcpy(y, ctx->mm_hy, yb);
    return 1;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    /* Shares ctx->x/y/gate/up with the grouped path; complete any in-flight submit
     * before reusing those buffers. */
    if (ctx->group_pending) coli_cuda_expert_group_finish(gate->device);
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,yb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    if (!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb) ||
        !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,yb)) return 0;
    std::memcpy(ctx->host_x,x,xb);
    if (!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),"expert input upload")) return 0;
    dim3 hidden_grid((unsigned)I,(unsigned)S), output_grid((unsigned)D,(unsigned)S);
    quant_matmul<<<hidden_grid,256,0,ctx->stream>>>(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D));
    quant_matmul<<<hidden_grid,256,0,ctx->stream>>>(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D));
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,n);
    quant_matmul<<<output_grid,256,0,ctx->stream>>>(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I));
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,yb,cudaMemcpyDeviceToHost,ctx->stream),"expert output download") ||
        !cuda_ok(cudaStreamSynchronize(ctx->stream),"expert MLP synchronize")) return 0;
    std::memcpy(y,ctx->host_y,yb);
    return 1;
}

/* Enqueue a grouped-expert call on the device stream (descriptor + input upload,
 * the gate/up/down/SiLU kernels, an async result download into pinned staging) but
 * do NOT synchronize. The caller runs CPU-side work (non-resident experts, shared
 * expert) while the GPU computes, then calls coli_cuda_expert_group_finish to sync
 * and land the result — turning the MoE expert phase from GPU_time + CPU_time into
 * max(GPU_time, CPU_time). Exactly one submit may be outstanding per device.
 *
 * The int4 fast path (COLI_CUDA_FAST_GEMV, default on) uses the warp-per-row fused
 * kernels; set it to 0 for the block-per-output path. */
extern "C" int coli_cuda_expert_group_submit(ColiCudaTensor *const *gates,
                                             ColiCudaTensor *const *ups,
                                             ColiCudaTensor *const *downs,
                                             const int *rows, int count,
                                             float *y, const float *x) {
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1 || count > 64) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    GroupDesc host[64];
    int all_s4=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(ctx->group_pending) return 0;   /* a prior submit was never finished */
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    if(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)) return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    ctx->pend_profile=0;
    if(profile){ ctx->pend_profile=1;
        for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ctx->pend_ev[i]),"profile event")){ ctx->pend_profile=0; break; } }
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                cudaMemcpyHostToDevice,ctx->stream),"expert group descriptors")) return 0;
    if(ctx->pend_profile) cudaEventRecord(ctx->pend_ev[0],ctx->stream);
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                "expert group input upload")) return 0;
    if(ctx->pend_profile) cudaEventRecord(ctx->pend_ev[1],ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    int tc=getenv("COLI_CUDA_TC_INT4")&&atoi(getenv("COLI_CUDA_TC_INT4"));
    tc=tc&&all_s4&&D%32==0&&I%32==0&&D%8==0&&I%8==0;
    int tc_min=getenv("COLI_CUDA_TC_MIN_ROWS")?atoi(getenv("COLI_CUDA_TC_MIN_ROWS")):8;
    for(int c=0;c<count&&tc;c++)tc=rows[c]>=tc_min;
    if(tc){
        size_t qb=(size_t)(total+7)*(size_t)(D>I?D:I)/2;
        if(!reserve_bytes((void**)&ctx->qx,&ctx->qx_cap,qb)||
           !reserve(&ctx->qscale,&ctx->qscale_cap,(size_t)(total+7)*sizeof(float)))return 0;
        cudaMemsetAsync(ctx->qx,0,qb,ctx->stream);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->x,total,D);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->gate,ctx->qx,ctx->qscale,dev,D,I,0);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->up,ctx->qx,ctx->qscale,dev,D,I,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->gate,total,I);
        grouped_s4_wmma<<<dim3((unsigned)((D+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->y,ctx->qx,ctx->qscale,dev,I,D,2);
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        int fast=!getenv("COLI_CUDA_FAST_GEMV")||atoi(getenv("COLI_CUDA_FAST_GEMV"));
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(fast){
            /* Warp-per-output row; SiLU(gate)*up folded into the hidden pass (ctx->gate
             * holds the product, ctx->up is unused), then a fused down projection. */
            dim3 hg((unsigned)((I+COLI_WPB-1)/COLI_WPB),(unsigned)max_rows,(unsigned)count);
            dim3 og((unsigned)((D+COLI_WPB-1)/COLI_WPB),(unsigned)max_rows,(unsigned)count);
            grouped_hidden_w4_dual_fused<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D);
            grouped_down_w4_fast<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
        }else{
            dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
            if(dual)grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
            else{
                grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
                grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            }
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
            grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
        }
    }else{
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
    if(ctx->pend_profile) cudaEventRecord(ctx->pend_ev[2],ctx->stream);
    if(!cuda_ok(cudaGetLastError(),"expert group launch")) return 0;
    if(!cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                "expert group output download")) return 0;
    if(ctx->pend_profile) cudaEventRecord(ctx->pend_ev[3],ctx->stream);
    ctx->group_pending=1; ctx->pend_y=y; ctx->pend_bytes=xb;
    ctx->pend_experts=(uint64_t)count; ctx->pend_rows=(uint64_t)total;
    return 1;
}

/* Sync the device stream, copy the pending result to the caller buffer recorded at
 * submit, and fold in profiling/stats. Safe to call with nothing outstanding. */
extern "C" int coli_cuda_expert_group_finish(int device) {
    DeviceContext *ctx=find_ctx(device); if(!ctx) return 0;
    if(!ctx->group_pending) return 1;
    ctx->group_pending=0;
    if(!select_ctx(ctx)) return 0;
    if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize")) return 0;
    std::memcpy(ctx->pend_y,ctx->host_y,ctx->pend_bytes);
    if(ctx->pend_profile){
        float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ctx->pend_ev[0],ctx->pend_ev[1]);
        cudaEventElapsedTime(&b,ctx->pend_ev[1],ctx->pend_ev[2]);
        cudaEventElapsedTime(&c,ctx->pend_ev[2],ctx->pend_ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ctx->pend_ev[i]);
        ctx->pend_profile=0;
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=ctx->pend_experts; g_group_rows+=ctx->pend_rows; }
    return 1;
}

/* Synchronous grouped call: submit then immediately finish. For callers that do not
 * overlap CPU work with the GPU (glm.c and the multi-device path). */
extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    if(!coli_cuda_expert_group_submit(gates,ups,downs,rows,count,y,x)) return 0;
    return coli_cuda_expert_group_finish(gates[0]->device);
}

/* JIT-streaming grouped experts: same kernels as coli_cuda_expert_group, but the
 * int4 weights are NOT resident. They are uploaded from host slabs into reusable
 * device scratch each call (this is the PCIe-bound path — measure vs CPU). All
 * experts must share shape (D,I) and be int4 (fmt==2). Host weights use the
 * offset-8 nibble encoding; offset_to_signed_s4 converts them in place after upload,
 * exactly as coli_cuda_tensor_upload does for resident tensors. */
extern "C" int coli_cuda_expert_group_stream(
        const void *const *gw,const void *const *uw,const void *const *dw,
        const float *const *gs,const float *const *us,const float *const *ds,
        const int *rows,int count,int D,int I,int device,
        float *y,const float *x){
    if(!gw||!uw||!dw||!gs||!us||!ds||!rows||!x||!y||count<1||count>64||D<1||I<1) return 0;
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    int total=0,max_rows=0;
    for(int c=0;c<count;c++){ if(rows[c]<1) return 0; total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c]; }
    size_t gstride=(size_t)I*((D+1)/2), dstride=(size_t)D*((I+1)/2);
    size_t gsz=(size_t)count*gstride, dsz=(size_t)count*dstride;
    size_t gssz=(size_t)count*I*sizeof(float), dssz=(size_t)count*D*sizeof(float);
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve_bytes(&ctx->sw_g,&ctx->sw_g_cap,gsz)||!reserve_bytes(&ctx->sw_u,&ctx->sw_u_cap,gsz)||
       !reserve_bytes(&ctx->sw_d,&ctx->sw_d_cap,dsz)||
       !reserve(&ctx->ss_g,&ctx->ss_g_cap,gssz)||!reserve(&ctx->ss_u,&ctx->ss_u_cap,gssz)||
       !reserve(&ctx->ss_d,&ctx->ss_d_cap,dssz)||
       !reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    GroupDesc host[64]; int off=0;
    for(int c=0;c<count;c++){
        uint8_t *gd=(uint8_t*)ctx->sw_g+(size_t)c*gstride;
        uint8_t *ud=(uint8_t*)ctx->sw_u+(size_t)c*gstride;
        uint8_t *dd=(uint8_t*)ctx->sw_d+(size_t)c*dstride;
        float *gsd=ctx->ss_g+(size_t)c*I,*usd=ctx->ss_u+(size_t)c*I,*dsd=ctx->ss_d+(size_t)c*D;
        if(!cuda_ok(cudaMemcpyAsync(gd,gw[c],gstride,cudaMemcpyHostToDevice,ctx->stream),"stream gate w")||
           !cuda_ok(cudaMemcpyAsync(ud,uw[c],gstride,cudaMemcpyHostToDevice,ctx->stream),"stream up w")||
           !cuda_ok(cudaMemcpyAsync(dd,dw[c],dstride,cudaMemcpyHostToDevice,ctx->stream),"stream down w")||
           !cuda_ok(cudaMemcpyAsync(gsd,gs[c],(size_t)I*sizeof(float),cudaMemcpyHostToDevice,ctx->stream),"stream gate s")||
           !cuda_ok(cudaMemcpyAsync(usd,us[c],(size_t)I*sizeof(float),cudaMemcpyHostToDevice,ctx->stream),"stream up s")||
           !cuda_ok(cudaMemcpyAsync(dsd,ds[c],(size_t)D*sizeof(float),cudaMemcpyHostToDevice,ctx->stream),"stream down s")) return 0;
        host[c]={gd,ud,dd,gsd,usd,dsd,2,2,2,rows[c],off}; off+=rows[c];
    }
    offset_to_signed_s4<<<(unsigned)((gsz+255)/256),256,0,ctx->stream>>>((uint8_t*)ctx->sw_g,gsz);
    offset_to_signed_s4<<<(unsigned)((gsz+255)/256),256,0,ctx->stream>>>((uint8_t*)ctx->sw_u,gsz);
    offset_to_signed_s4<<<(unsigned)((dsz+255)/256),256,0,ctx->stream>>>((uint8_t*)ctx->sw_d,dsz);
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice,ctx->stream),"stream desc")||
       !cuda_ok(cudaMemcpyAsync(ctx->x,x,xb,cudaMemcpyHostToDevice,ctx->stream),"stream x")) return 0;
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    if(!getenv("COLI_CUDA_FAST_GEMV")||atoi(getenv("COLI_CUDA_FAST_GEMV"))){
        dim3 hg((unsigned)((I+COLI_WPB-1)/COLI_WPB),(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)((D+COLI_WPB-1)/COLI_WPB),(unsigned)max_rows,(unsigned)count);
        grouped_hidden_w4_dual_fused<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D);
        grouped_down_w4_fast<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else{
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
    if(!cuda_ok(cudaGetLastError(),"stream expert launch")||
       !cuda_ok(cudaMemcpyAsync(y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),"stream expert download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"stream expert synchronize")) return 0;
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
    return 1;
}

extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w,float *ctx,const float *q,
                                            const float *latent,const float *rope,int H,int Q,
                                            int R,int V,int K,int T,float scale){
    if(!w||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention rope upload"))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    attention_absorb_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,w->weights,w->scales,
        w->fmt,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"attention absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_gqa_attention(float *ctx,const float *q,const float *k_cache,const float *v_cache,
                                       int S,int H,int Hkv,int hd,int st0,int pos_base,int max_t,int device){
    if(!ctx||!q||!k_cache||!v_cache||S<1||H<1||Hkv<1||hd<1||max_t<1||H%Hkv)return 0;
    int nrep=H/Hkv,max_pos=pos_base+S-1,nt=max_pos+1-st0;
    if(nt<1||nt>GQA_MAX_NT)return 0;
    DeviceContext *dc=find_ctx(device); if(!select_ctx(dc)) return 0;
    float scale=1.f/sqrtf((float)hd);
    size_t qb=(size_t)S*H*hd*sizeof(float),kb=(size_t)Hkv*max_t*hd*sizeof(float);
    size_t cb=qb,sb=(size_t)S*H*GQA_MAX_NT*sizeof(float);
    if(!reserve(&dc->gq,&dc->gq_cap,qb)||!reserve(&dc->gk,&dc->gk_cap,kb)||
       !reserve(&dc->gv,&dc->gv_cap,kb)||!reserve(&dc->gctx,&dc->gctx_cap,cb)||
       !reserve(&dc->gsc,&dc->gsc_cap,sb)) return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->gq,q,qb,cudaMemcpyHostToDevice,dc->stream),"gqa q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->gk,k_cache,kb,cudaMemcpyHostToDevice,dc->stream),"gqa k upload")||
       !cuda_ok(cudaMemcpyAsync(dc->gv,v_cache,kb,cudaMemcpyHostToDevice,dc->stream),"gqa v upload")) return 0;
    dim3 grid(H,S);
    gqa_attn_kernel<<<grid,256,0,dc->stream>>>(dc->gctx,dc->gq,dc->gk,dc->gv,dc->gsc,S,H,Hkv,hd,st0,pos_base,max_t,scale,nrep);
    if(!cuda_ok(cudaGetLastError(),"gqa attention launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->gctx,cb,cudaMemcpyDeviceToHost,dc->stream),"gqa ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"gqa attention synchronize")) return 0;
    return 1;
}

/* Ensure the resident fp16 KV cache for `layer` matches (Hkv,max_t,hd) on `device`,
 * (re)allocating if the geometry changed. Returns the cache or nullptr on failure. */
static KVCache *kv_cache_ensure(int layer,int Hkv,int max_t,int hd,int device){
    if(layer<0||layer>=COLI_CUDA_MAX_LAYERS) return nullptr;
    KVCache *c=&g_kv[layer];
    if(c->k&&(c->Hkv!=Hkv||c->max_t!=max_t||c->hd!=hd||c->device!=device)){
        cudaFree(c->k); cudaFree(c->v); c->k=c->v=nullptr;
    }
    if(!c->k){
        size_t bytes=(size_t)Hkv*max_t*hd*sizeof(__half);
        if(!cuda_ok(cudaMalloc(&c->k,bytes),"kv cache K alloc")) return nullptr;
        if(!cuda_ok(cudaMalloc(&c->v,bytes),"kv cache V alloc")){ cudaFree(c->k); c->k=nullptr; return nullptr; }
        c->Hkv=Hkv; c->max_t=max_t; c->hd=hd; c->device=device;
    }
    return c;
}

/* Resident-cache GQA attention. Appends the S new K/V rows (k_new/v_new, [S,Hkv,hd])
 * to the per-layer fp16 cache, then runs attention over the whole resident cache —
 * so only the new row(s), q and ctx cross PCIe each token, not the entire cache.
 * Falls back (returns 0) so the caller can use the CPU path on any failure. */
extern "C" int coli_cuda_gqa_attention_cached(int layer,float *ctx,const float *q,
                                       const float *k_new,const float *v_new,
                                       int S,int H,int Hkv,int hd,int st0,int pos_base,int max_t,int device){
    if(!ctx||!q||!k_new||!v_new||S<1||H<1||Hkv<1||hd<1||max_t<1||H%Hkv)return 0;
    int nrep=H/Hkv,max_pos=pos_base+S-1,nt_stride=max_pos+1-st0;
    if(nt_stride<1)return 0;
    DeviceContext *dc=find_ctx(device); if(!select_ctx(dc)) return 0;
    KVCache *kc=kv_cache_ensure(layer,Hkv,max_t,hd,device); if(!kc) return 0;
    float scale=1.f/sqrtf((float)hd);
    size_t qb=(size_t)S*H*hd*sizeof(float), nb=(size_t)S*Hkv*hd*sizeof(float);
    size_t cb=qb, sb=(size_t)S*H*nt_stride*sizeof(float);
    /* dc->gk/gv now stage only the S new rows (float) before the append kernel. */
    if(!reserve(&dc->gq,&dc->gq_cap,qb)||!reserve(&dc->gk,&dc->gk_cap,nb)||
       !reserve(&dc->gv,&dc->gv_cap,nb)||!reserve(&dc->gctx,&dc->gctx_cap,cb)||
       !reserve(&dc->gsc,&dc->gsc_cap,sb)) return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->gq,q,qb,cudaMemcpyHostToDevice,dc->stream),"gqa q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->gk,k_new,nb,cudaMemcpyHostToDevice,dc->stream),"gqa k append upload")||
       !cuda_ok(cudaMemcpyAsync(dc->gv,v_new,nb,cudaMemcpyHostToDevice,dc->stream),"gqa v append upload")) return 0;
    kv_append_kernel<<<dim3((unsigned)S,(unsigned)Hkv),256,0,dc->stream>>>(
        kc->k,kc->v,dc->gk,dc->gv,S,Hkv,hd,pos_base,max_t);
    dim3 grid(H,S);
    gqa_attn_half_kernel<<<grid,256,0,dc->stream>>>(dc->gctx,dc->gq,kc->k,kc->v,dc->gsc,
        S,H,Hkv,hd,st0,pos_base,max_t,nt_stride,scale,nrep);
    if(!cuda_ok(cudaGetLastError(),"gqa cached attention launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->gctx,cb,cudaMemcpyDeviceToHost,dc->stream),"gqa cached ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"gqa cached attention synchronize")) return 0;
    return 1;
}

/* Free all resident KV caches (called from shutdown). */
static void kv_cache_free_all(void){
    for(int i=0;i<COLI_CUDA_MAX_LAYERS;i++){ KVCache *c=&g_kv[i];
        if(c->k){ cudaSetDevice(c->device); cudaFree(c->k); }
        if(c->v) cudaFree(c->v);
        c->k=c->v=nullptr; c->Hkv=c->max_t=c->hd=0; c->device=0; }
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = coli_cuda_tensor_bytes(tensor);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->slab) cudaFree(tensor->slab);        /* owner: one free covers all sections */
    else if (!tensor->borrowed) {
        if (tensor->weights) cudaFree(tensor->weights);
        if (tensor->scales) cudaFree(tensor->scales);
    }
    std::free(tensor);
}

/* Slab owners report the whole allocation; borrowers report 0 so summing a
 * g/u/d triple never double-counts. */
extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    if (tensor->slab) return tensor->slab_bytes;
    if (tensor->borrowed) return 0;
    return tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}
