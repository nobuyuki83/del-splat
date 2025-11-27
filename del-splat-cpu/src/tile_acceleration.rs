use num_traits::AsPrimitive;

pub fn tile2pnt<PNT2AABBDEPTH, INDEX>(
    num_pnt: usize,
    pnt2aabbdepth: PNT2AABBDEPTH,
    img_shape: (usize, usize),
    tile_size: usize,
) -> (Vec<INDEX>, Vec<INDEX>)
where
    INDEX: num_traits::PrimInt + std::ops::AddAssign<INDEX> + AsPrimitive<usize>,
    usize: AsPrimitive<INDEX>,
    PNT2AABBDEPTH: Fn(usize) -> ([f32; 4], f32),
{
    let tile_shape = (
        img_shape.0 / tile_size + if img_shape.0 % tile_size == 0 { 0 } else { 1 },
        img_shape.1 / tile_size + if img_shape.1 % tile_size == 0 { 0 } else { 1 },
    );
    let num_tile = tile_shape.0 * tile_shape.1;
    let mut tile2ind = vec![INDEX::zero(); num_tile + 1];
    for i_vtx in 0..num_pnt {
        let (aabb2, _depth) = pnt2aabbdepth(i_vtx);
        let tiles = del_geo_core::aabb2::overlapping_tiles(&aabb2, tile_size, tile_shape);
        for &i_tile in tiles.iter() {
            tile2ind[i_tile + 1] += INDEX::one();
        }
    }
    for i_tile in 0..num_tile {
        let ind0 = tile2ind[i_tile + 1];
        tile2ind[i_tile + 1] = tile2ind[i_tile] + ind0;
    }
    let num_ind: usize = tile2ind[num_tile].as_();
    let mut ind2vtx = vec![INDEX::zero(); num_ind];
    let mut ind2tiledepth = Vec::<(usize, usize, f32)>::with_capacity(num_ind);
    for i_vtx in 0..num_pnt {
        let (aabb2, depth) = pnt2aabbdepth(i_vtx);
        let mut depth = depth + 1f32;
        if depth < 0f32 {
            depth = 0f32;
        }
        let tiles = del_geo_core::aabb2::overlapping_tiles(&aabb2, tile_size, tile_shape);
        for &i_tile in tiles.iter() {
            ind2vtx[ind2tiledepth.len()] = i_vtx.as_();
            ind2tiledepth.push((i_vtx, i_tile, depth));
        }
    }
    assert_eq!(ind2tiledepth.len(), num_ind);
    ind2tiledepth.sort_by(|&a, &b| {
        if a.1 == b.1 {
            a.2.partial_cmp(&b.2).unwrap()
        } else {
            a.1.cmp(&b.1)
        }
    });
    for iind in 0..ind2tiledepth.len() {
        ind2vtx[iind] = ind2tiledepth[iind].0.as_();
    }
    (tile2ind, ind2vtx)
}

pub fn tile2pnt_gauss<INDEX>(
    pnt2pixxyndcz: &[f32],
    pnt2pixaabb: &[f32],
    img_shape: (usize, usize),
    tile_size: usize,
) -> (Vec<INDEX>, Vec<INDEX>)
where
    INDEX: num_traits::PrimInt + std::ops::AddAssign<INDEX> + AsPrimitive<usize>,
    usize: AsPrimitive<INDEX>,
{
    let num_pnt = pnt2pixxyndcz.len() / 3;
    assert_eq!(pnt2pixaabb.len(), num_pnt * 4);
    let point2aabbdepth = |i_point: usize| {
        let aabb = arrayref::array_ref![pnt2pixaabb, i_point * 4, 4].to_owned();
        (aabb, pnt2pixxyndcz[i_point * 3 + 2])
    };
    tile2pnt::<_, INDEX>(num_pnt, point2aabbdepth, img_shape, tile_size)
}

pub fn tile2pnt_circle<INDEX>(
    pnt2pixxyndcz: &[f32],
    pnt2pixrad: &[f32],
    img_shape: (usize, usize),
    tile_size: usize,
) -> (Vec<INDEX>, Vec<INDEX>)
where
    INDEX: num_traits::PrimInt + std::ops::AddAssign<INDEX> + AsPrimitive<usize>,
    usize: AsPrimitive<INDEX>,
{
    let num_pnt = pnt2pixxyndcz.len() / 3;
    assert_eq!(pnt2pixrad.len(), num_pnt);
    let point2aabbdepth = |i_point: usize| {
        let pixco = arrayref::array_ref![pnt2pixxyndcz, i_point * 3, 2];
        let aabb = del_geo_core::aabb2::from_point(pixco, pnt2pixrad[i_point]);
        (aabb, pnt2pixxyndcz[i_point * 3 + 2])
    };
    tile2pnt::<_, INDEX>(num_pnt, point2aabbdepth, img_shape, tile_size)
}

pub fn check_tile2pnt_circle(
    pnt2pixxydepth: &[f32],
    pnt2pixrad: &[f32],
    tile_shape: (usize, usize),
    tile_size: usize,
    tile2idx: &[u32],
    idx2pnt_gpu: &[u32],
) {
    let num_pnt = pnt2pixxydepth.len() / 3;
    let num_ind = {
        // debug tile2ind using cpu
        // let pnt2splat = dev.dtoh_sync_copy(&pnt2splat_dev)?;
        let num_tile = tile_shape.0 * tile_shape.1;
        let mut tile2idx_cpu = vec![0usize; num_tile + 1];
        for i_vtx in 0..num_pnt {
            let p0 = arrayref::array_ref![pnt2pixxydepth, i_vtx * 3, 2];
            let rad = pnt2pixrad[i_vtx];
            let aabb2 = del_geo_core::aabb2::from_point(p0, rad);
            let tiles = del_geo_core::aabb2::overlapping_tiles(&aabb2, tile_size, tile_shape);
            for &i_tile in tiles.iter() {
                tile2idx_cpu[i_tile + 1] += 1;
            }
        }
        for i_tile in 0..num_tile {
            tile2idx_cpu[i_tile + 1] += tile2idx_cpu[i_tile];
        }
        // let tile2idx = dev.dtoh_sync_copy(&tile2idx_dev)?;
        tile2idx
            .iter()
            .zip(tile2idx_cpu.iter())
            .for_each(|(&a, &b)| {
                assert_eq!(a as usize, b);
            });
        tile2idx[tile_shape.0 * tile_shape.1]
    }; // end debug tile2ind
    {
        // assert ind2pnt by cpu code
        //let num_ind = idx2pnt_dev.len();
        //let pnt2splat = dev.dtoh_sync_copy(&pnt2splat_dev)?;
        //let idx2pnt_gpu = dev.dtoh_sync_copy(&idx2pnt_dev)?;
        let mut idx2pnt_cpu = vec![0usize; num_ind as usize];
        let mut ind2tiledepth = Vec::<(usize, usize, f32)>::with_capacity(num_ind as usize);
        for i_pnt in 0..num_pnt {
            let pos_pix = arrayref::array_ref![pnt2pixxydepth, i_pnt * 3, 2];
            let rad_pix = pnt2pixrad[i_pnt];
            let depth = pnt2pixxydepth[i_pnt * 3 + 2] + 1f32;
            let aabb2 = del_geo_core::aabb2::from_point(pos_pix, rad_pix);
            let tiles = del_geo_core::aabb2::overlapping_tiles(&aabb2, tile_size, tile_shape);
            for &i_tile in tiles.iter() {
                idx2pnt_cpu[ind2tiledepth.len()] = i_pnt;
                ind2tiledepth.push((i_pnt, i_tile, depth));
            }
        }
        assert_eq!(ind2tiledepth.len(), num_ind as usize);
        ind2tiledepth.sort_by(|&a, &b| {
            if a.1 == b.1 {
                a.2.partial_cmp(&b.2).unwrap()
            } else {
                a.1.cmp(&b.1)
            }
        });
        for iind in 0..ind2tiledepth.len() {
            idx2pnt_cpu[iind] = ind2tiledepth[iind].0;
        }
        assert_eq!(idx2pnt_cpu.len(), idx2pnt_gpu.len());
        idx2pnt_cpu
            .iter()
            .zip(idx2pnt_gpu.iter())
            .for_each(|(&ipnt0, &ipnt1)| {
                assert_eq!(ipnt0, ipnt1 as usize);
            })
    } // assert "idx2pnt" using cpu
}
