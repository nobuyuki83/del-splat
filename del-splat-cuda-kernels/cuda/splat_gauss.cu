#include "del_geo/mat2_sym.h"
#include "del_geo/mat4_col_major.h"
#include "del_geo/mat2x3_col_major.h"
#include "del_geo/quaternion.h"
#include "del_geo/aabb2.h"
#include "tile_acceleration.h"

extern "C" {

struct Splat3 {
    float xyz[3];
    float rgb_dc[3];
    float rgb_sh[45];
    float opacity;
    float scale[3];
    float quaternion[4];
};

struct Splat2 {
    float pos_pix[2];
    float sig_inv[3];
    float aabb[4];
    float rgb[3];
    float alpha;
    float ndc_z;
};

__global__
void splat3_to_splat2(
  uint32_t num_pnt,
  float* pnt2pixxydepth,
  float* pnt2pixconvinv,
  float* pnt2pixaabb,
  const float* pnt2xyz,
  const float* pnt2quat,
  const float* pnt2scale,
  const float *transform_world2ndc,
  const uint32_t img_w,
  const uint32_t img_h)
{
    int i_pnt = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_pnt >= num_pnt ){ return; }
    //
    const float* xyz = pnt2xyz + i_pnt * 3;
    const cuda::std::array<float,9> world2ndc = mat4_col_major::jacobian_transform(transform_world2ndc, xyz);
    const cuda::std::array<float,6> ndc2pix = mat2x3_col_major::transform_ndc2pix(img_w, img_h);
    const cuda::std::array<float,6> world2pix = mat2x3_col_major::mult_mat3_col_major(ndc2pix.data(), world2ndc.data());
    const auto pos_ndc = mat4_col_major::transform_homogeneous(
        transform_world2ndc, xyz);
    const float pos_scrn[3] = {pos_ndc[0], pos_ndc[1], 1.f};
    const auto pixxy = mat2x3_col_major::mult_vec3(ndc2pix.data(), pos_scrn);
    const cuda::std::array<float,3> pixconv = mat2_sym::projected_spd_mat3(
        world2pix.data(),
        pnt2quat + i_pnt * 4,
        pnt2scale + i_pnt * 3);
    const cuda::std::array<float,3> pixconvinv = mat2_sym::safe_inverse_preserve_positive_definiteness(pixconv.data(), 1.0e-5f);
    const cuda::std::array<float,4> _aabb0 = mat2_sym::aabb2(pixconvinv.data());
    const cuda::std::array<float,4> _aabb1 = aabb2::scale(_aabb0.data(), 3.f);
    const cuda::std::array<float,4> pixaabb = aabb2::translate(_aabb1.data(), pixxy.data());
    //
    pnt2pixxydepth[i_pnt*3+0] = pixxy[0];
    pnt2pixxydepth[i_pnt*3+1] = pixxy[1];
    pnt2pixxydepth[i_pnt*3+2] = pos_ndc[2];
    pnt2pixconvinv[i_pnt*3+0] = pixconvinv[0];
    pnt2pixconvinv[i_pnt*3+1] = pixconvinv[1];
    pnt2pixconvinv[i_pnt*3+2] = pixconvinv[2];
    pnt2pixaabb[i_pnt*4+0] = pixaabb[0];
    pnt2pixaabb[i_pnt*4+1] = pixaabb[1];
    pnt2pixaabb[i_pnt*4+2] = pixaabb[2];
    pnt2pixaabb[i_pnt*4+3] = pixaabb[3];
}


__global__
void rasterize_splat_using_tile(
    uint32_t img_w,
    uint32_t img_h,
    float* d_pix2rgb,
    uint32_t tile_w,
    uint32_t tile_h,
    uint32_t tile_size,
    const uint32_t* d_tile2idx,
    const uint32_t* d_idx2pnt,
    const float* pnt2pixxydepth,
    const float* pnt2pixrad,
    const float* pnt2pixaabb,
    const float* pnt2pixconvinv,
    const float* pnt2oppacity,
    const float* pnt2rgb)
{
    const uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    if( ix >= img_w ){ return; }
    //
    const uint32_t iy = blockDim.y * blockIdx.y + threadIdx.y;
    if( iy >= img_h ){ return; }
    //
    const uint32_t i_tile = (iy / tile_size) * tile_w + (ix / tile_size);
    const float t[2] = {float(ix) + 0.5f, float(iy) + 0.5f};
    float alpha_sum = 0.f;
    float alpha_occu = 1.f;
    // iterate front (z large) to back (z small)
    const uint32_t num_pnt = d_tile2idx[i_tile+1] - d_tile2idx[i_tile];
    for (uint32_t iidx=0;iidx<num_pnt;++iidx) {
        uint32_t idx = d_tile2idx[i_tile] + num_pnt - 1 - iidx;
        const uint32_t i_pnt = d_idx2pnt[idx];
        const float* pixxy = pnt2pixxydepth + i_pnt * 3;
        const float* pixaabb = pnt2pixaabb + i_pnt * 4;
        const float* pixconvinv = pnt2pixconvinv + i_pnt * 3;
        const float oppacity = pnt2oppacity[i_pnt];
        const float* rgb = pnt2rgb + i_pnt * 3;
        // front to back
        if( !aabb2::is_inlcude_point(pixaabb, t) ){
            continue;
        }
        const float t0[2] = {t[0] - pixxy[0], t[1] - pixxy[1]};
        float _e = mat2_sym::mult_vec_from_both_sides(pixconvinv, t0, t0);
        float e = expf(-0.5 * _e) * oppacity;
        float e_out = alpha_occu * e;
        d_pix2rgb[(iy * img_w + ix) * 3 + 0] += rgb[0] * e_out;
        d_pix2rgb[(iy * img_w + ix) * 3 + 1] += rgb[1] * e_out;
        d_pix2rgb[(iy * img_w + ix) * 3 + 2] += rgb[2] * e_out;
        alpha_occu *= 1.f - e;
        alpha_sum += e_out;
        if( alpha_sum > 0.999 ){
            break;
        }
    }
}

__global__
void count_splat_in_tile(
  uint32_t num_pnt,
  const Splat2* pnt2splat,
  uint32_t* tile2ind,
  uint32_t* pnt2ind,
  uint32_t tile_w,
  uint32_t tile_h,
  uint32_t tile_size)
{
    int i_pnt = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_pnt >= num_pnt ){ return; }
    //
    const Splat2& splat = pnt2splat[i_pnt];
    const float* aabb = splat.aabb;
    const cuda::std::array<float,4> aabb0 {aabb[0], aabb[1], aabb[2], aabb[3]};
    //
    tile_acceleration::count_splat_in_tile(
        i_pnt, aabb0,
        tile2ind, pnt2ind,
        tile_w, tile_h, tile_size);
}

__global__
void fill_index_info(
  uint32_t num_pnt,
  const Splat2* pnt2splat,
  const uint32_t* pnt2idx,
  uint64_t* idx2tiledepth,
  uint32_t* idx2pnt,
  uint32_t tile_w,
  uint32_t tile_h,
  uint32_t tile_size)
{
    int i_pnt = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_pnt >= num_pnt ){ return; }
    //
    const Splat2& splat = pnt2splat[i_pnt];
    const float* aabb = splat.aabb;
    const cuda::std::array<float,4> aabb0 {aabb[0], aabb[1], aabb[2], aabb[3]};
    tile_acceleration::fill_index_info(
        i_pnt, aabb0, splat.ndc_z,
        pnt2idx, idx2tiledepth, idx2pnt,
        tile_w, tile_h, tile_size);
}



}