__device__
void ztest_and_write_color(
    int i_pix,
    float ndcz_new,
    const float* rgb,
    float* pix2unitz,
    float* pix2rgb)
{
    // depthBuf[idx] を uint32 として扱う
    unsigned int* depthIntPtr =
        reinterpret_cast<unsigned int*>(&pix2unitz[i_pix]);

    // 現在値を読み込み
    unsigned int oldBits = *depthIntPtr;
    float oldDepth = __uint_as_float(oldBits);

    // newDepth が既に負けているなら何もしない
    if (ndcz_new >= oldDepth) return;

    while (true) {
        // 現在の値に対して自分の newDepth を書き込もうとする
        unsigned int newBits = __float_as_uint(ndcz_new);

        // oldBits が変わっていなければ newBits に置換される
        unsigned int prev =
            atomicCAS(depthIntPtr, oldBits, newBits);

        if (prev == oldBits) {
            // ★ 自分が depth を更新することに成功したので勝ち
            //    このスレッドだけ color を書き込む
            pix2rgb[i_pix*3+0] = rgb[0];
            pix2rgb[i_pix*3+1] = rgb[1];
            pix2rgb[i_pix*3+2] = rgb[2];
            break;
        }

        // 途中で別スレッドが書き換えたので、再度比較する
        oldBits = prev;
        oldDepth = __uint_as_float(oldBits);

        // すでに自分より手前の depth が入っていれば諦める
        if (ndcz_new >= oldDepth) {
            break;
        }
        // まだ自分の方が手前なら、再び CAS を試行
    }
}

extern "C" __global__
void rasterize_zbuffer(
    uint32_t num_pnt,
    const float* pnt2pixxyndcz,
    const float* pnt2rgb,
    float* pix2unitdepth,
    float* pix2rgb,
    int width,
    int height)
{
    int i_pnt = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_pnt >= num_pnt) return;
    //
    const float* pixxy = pnt2pixxyndcz + i_pnt * 3;
    const float* rgb = pnt2rgb + i_pnt * 3;
    const float ndcz = pnt2pixxyndcz[i_pnt*3+2];
    if (ndcz<-1. || ndcz>1.){ return; }
    const float unitdepth = 1.0-(ndcz + 1.)*0.5; // large z -> small depth -> near to the viewer
    int i_x = static_cast<uint32_t>(pixxy[0]);
    int i_y = static_cast<uint32_t>(pixxy[1]);
    if (i_x < 0 || i_x >= width ){ return; }
    if (i_y < 0 || i_y >= height ){ return; }
    int i_pix = i_y * width + i_x;
    ztest_and_write_color(i_pix, unitdepth, rgb,
                          pix2unitdepth,
                          pix2rgb);
}