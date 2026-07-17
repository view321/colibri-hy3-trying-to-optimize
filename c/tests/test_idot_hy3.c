/* Exactness test for hy3.c's copies of the integer-dot kernels (same contract as
 * test_idot.c, which covers glm.c's). The two engines carry the kernels by copy,
 * and hy3's AVX2 dot_i8i8 shipped with sign(x,x) — computing sum(|w|*|x|) instead
 * of sum(w*x) — precisely because no test compiled hy3.c's copy. This one does.
 *
 * Covers: odd sizes (scalar tail), sizes below one vector, the w=-128 edge
 * (sign-trick kernels must treat |−128| as 128 unsigned, not saturate to 127),
 * and random data at qrow_i8's contract (|x| <= 127, w full int8 range). */
#define main coli_hy3_main_unused
#include "../hy3.c"
#undef main

static uint32_t rng_state=0x12345678u;
static uint32_t xr(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }

static int32_t ref_i8i8(const int8_t *w, const int8_t *x, int I){
    int64_t s=0; for(int i=0;i<I;i++) s+=(int32_t)w[i]*x[i]; return (int32_t)s;
}
static int32_t ref_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int64_t s=0;
    for(int i=0;i<I;i++){ uint8_t b=w4[i>>1]; int v=(i&1)?((int)(b>>4)-8):((int)(b&0xF)-8); s+=v*x[i]; }
    return (int32_t)s;
}

/* Reference copy of the ORIGINAL scalar qrow_i8 — the vectorized one must match
 * bit-for-bit (scale float bits AND every quantized byte). */
static float ref_qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f)s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv); return s;
}

int main(void){
    static const int sizes[]={1,2,15,16,17,31,32,33,63,64,65,100,127,128,1408,4096,4097};
    static int8_t w[8192], x[8192]; static uint8_t w4[4096];
    static int8_t x4[4*8192];
    for(unsigned t=0;t<sizeof(sizes)/sizeof(sizes[0]);t++){
        int I=sizes[t];
        for(int rep=0;rep<64;rep++){
            for(int i=0;i<I;i++) x[i]=(int8_t)((int)(xr()%255)-127);      /* [-127,127]: qrow_i8's contract */
            for(int i=0;i<I;i++) w[i]=(int8_t)((int)(xr()%256)-128);      /* [-128,127]: full range */
            if(rep==0) for(int i=0;i<I;i++) w[i]=-128;                    /* sign-trick edge case */
            if(rep==1) for(int i=0;i<I;i++){ w[i]=127; x[i]=(int8_t)(i&1?-127:127); }
            for(int i=0;i<(I+1)/2;i++) w4[i]=(uint8_t)(xr()&0xFF);
            int32_t got=dot_i8i8(w,x,I), want=ref_i8i8(w,x,I);
            if(got!=want){ fprintf(stderr,"FAIL dot_i8i8 I=%d rep=%d: %d != %d\n",I,rep,got,want); return 1; }
            got=dot_i4i8(w4,x,I); want=ref_i4i8(w4,x,I);
            if(got!=want){ fprintf(stderr,"FAIL dot_i4i8 I=%d rep=%d: %d != %d\n",I,rep,got,want); return 1; }
            /* 4-row blocked kernels vs 4 independent reference dots */
            for(int i=0;i<4*I;i++) x4[i]=(int8_t)((int)(xr()%255)-127);
            int32_t o8[4], o4[4];
            dotrow_i8i8_x4(w,x4,I,o8); dotrow_i4i8_x4(w4,x4,I,o4);
            for(int k=0;k<4;k++){
                if(o8[k]!=ref_i8i8(w,x4+(int64_t)k*I,I)){
                    fprintf(stderr,"FAIL dotrow_i8i8_x4 I=%d rep=%d row=%d\n",I,rep,k); return 1; }
                if(o4[k]!=ref_i4i8(w4,x4+(int64_t)k*I,I)){
                    fprintf(stderr,"FAIL dotrow_i4i8_x4 I=%d rep=%d row=%d\n",I,rep,k); return 1; }
            }
        }
    }
    /* blocked matmul wrappers vs the same math done row-by-row (bit-exact floats:
     * identical operation order (float)dot*sc*sx per element on both sides) */
    { static float xf[9*4097], y[9*33], yref[9*33]; static int8_t xq[9*4097]; static float sx[9];
      static int8_t q8[33*4097]; static uint8_t q4[33*2049]; static float sc[33];
      static const int Ss[]={1,2,3,4,5,8,9};
      int O=33;
      for(unsigned ti=0;ti<sizeof(sizes)/sizeof(sizes[0]);ti+=3){
        int I=sizes[ti+2<17?ti+2:16]; if(I>4097) I=4097;
        for(int i=0;i<O*I;i++) q8[i]=(int8_t)((int)(xr()%256)-128);
        for(int i=0;i<O*((I+1)/2);i++) q4[i]=(uint8_t)(xr()&0xFF);
        for(int o=0;o<O;o++) sc[o]=((int)(xr()%1000)+1)/500.0f;
        for(unsigned si=0;si<sizeof(Ss)/sizeof(Ss[0]);si++){
            int S=Ss[si];
            for(int i=0;i<S*I;i++) xf[i]=((int)(xr()%2001)-1000)/997.0f;
            for(int s=0;s<S;s++) sx[s]=qrow_i8(xf+(int64_t)s*I,xq+(int64_t)s*I,I);
            matmul_q_idot(y,xq,sx,q8,sc,S,I,O);
            for(int s=0;s<S;s++) for(int o=0;o<O;o++)
                yref[(int64_t)s*O+o]=(float)ref_i8i8(q8+(int64_t)o*I,xq+(int64_t)s*I,I)*sc[o]*sx[s];
            for(int i=0;i<S*O;i++) if(y[i]!=yref[i]){
                fprintf(stderr,"FAIL matmul_q_idot I=%d S=%d idx=%d\n",I,S,i); return 1; }
            matmul_i4_idot(y,xq,sx,q4,sc,S,I,O);
            for(int s=0;s<S;s++) for(int o=0;o<O;o++)
                yref[(int64_t)s*O+o]=(float)ref_i4i8(q4+(int64_t)o*((I+1)/2),xq+(int64_t)s*I,I)*sc[o]*sx[s];
            for(int i=0;i<S*O;i++) if(y[i]!=yref[i]){
                fprintf(stderr,"FAIL matmul_i4_idot I=%d S=%d idx=%d\n",I,S,i); return 1; }
        }
      }
    }
    /* vectorized qrow_i8 vs the original scalar: scale AND bytes bit-exact */
    { static float xf[4097]; static int8_t qa[4097], qb[4097];
      for(unsigned t=0;t<sizeof(sizes)/sizeof(sizes[0]);t++){
        int I=sizes[t];
        for(int rep=0;rep<16;rep++){
            for(int i=0;i<I;i++) xf[i]=((int)(xr()%200001)-100000)/91.0f;
            if(rep==0) for(int i=0;i<I;i++) xf[i]=0.0f;              /* s -> 1e-12 guard */
            if(rep==1) xf[I-1]=3.14e8f;                              /* single dominant max */
            float sa=qrow_i8(xf,qa,I), sb=ref_qrow_i8(xf,qb,I);
            if(memcmp(&sa,&sb,4)||memcmp(qa,qb,(size_t)I)){
                fprintf(stderr,"FAIL qrow_i8 I=%d rep=%d (scale %a vs %a)\n",I,rep,sa,sb); return 1; }
        }
      }
    }
    printf("hy3 idot kernel exactness incl. x4/qrow (%s): ok\n", IDOT_KERNEL);
    return 0;
}
