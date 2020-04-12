from datetime import datetime
import halide as hl
from utils import time_diff, Point, box_down2, idx_layer, idx_im, idx_0, idx_1, tile_0, tile_1, TILE_SIZE, \
    MINIMUM_OFFSET, MAXIMUM_OFFSET
import math

'''
Merges images in the temporal dimension.
Weights different frames based on the similarity between the reference tile
and the alternate tiles, minimizing L1 distances (least absolute deviation).
Distances greater than some threshold (max_distribution) are discarded.

images : Halide buffer
    Burst frames to be merged
alignment : Halide function
    Calculated alignment of the burst frames

Returns: Halide buffer (merged image)
'''
def merge_temporal(images, alignment):
    weight = hl.Func("merge_temporal_weights")
    total_weight = hl.Func("merge_temporal_total_weights")
    output = hl.Func("merge_temporal_output")

    ix, iy, tx, ty, n = hl.Var('ix'), hl.Var('iy'), hl.Var('tx'), hl.Var('ty'), hl.Var('n')
    rdom0 = hl.RDom([(0, 16), (0, 16)])

    rdom1 = hl.RDom([(1, images.dim(2).extent() - 1)])

    imgs_mirror = hl.BoundaryConditions.mirror_interior(images, [(0, images.width()), (0, images.height())])

    layer = box_down2(imgs_mirror, "merge_layer")

    offset = Point(alignment[tx, ty, n]).clamp(Point(MINIMUM_OFFSET, MINIMUM_OFFSET),
                                               Point(MAXIMUM_OFFSET, MAXIMUM_OFFSET))

    al_x = idx_layer(tx, rdom0.x) + offset.x / 2
    al_y = idx_layer(ty, rdom0.y) + offset.y / 2

    ref_val = layer[idx_layer(tx, rdom0.x), idx_layer(ty, rdom0.y), 0]
    alt_val = layer[al_x, al_y, n]

    factor = 8.0
    min_distribution = 10
    max_distribution = 300

    distribution = hl.sum(hl.abs(hl.cast(hl.Int(32), ref_val) - hl.cast(hl.Int(32), alt_val))) / 256

    normal_distribution = hl.max(1, hl.cast(hl.Int(32), distribution) / factor - min_distribution / factor)

    weight[tx, ty, n] = hl.select(normal_distribution > (max_distribution - max_distribution), 0.0,
                                  1.0 / normal_distribution)

    total_weight[tx, ty] = hl.sum(weight[tx, ty, rdom1]) + 1

    offset = Point(alignment[tx, ty, rdom1])

    al_x = idx_im(tx, ix) + offset.x
    al_y = idx_im(ty, iy) + offset.y

    ref_val = imgs_mirror[idx_im(tx, ix), idx_im(ty, iy), 0]
    alt_val = imgs_mirror[al_x, al_y, rdom1]

    output[ix, iy, tx, ty] = hl.sum(weight[tx, ty, rdom1] * alt_val / total_weight[tx, ty]) + ref_val / total_weight[
        tx, ty]

    weight.compute_root().parallel(ty).vectorize(tx, 16)

    total_weight.compute_root().parallel(ty).vectorize(tx, 16)

    output.compute_root().parallel(ty).vectorize(ix, 32)

    return output


'''
Merges images in the spatial dimension.
Smoothly blends overlapping tiles using modified raised cosine window.

input : Halide buffer
    image (burst frames merged in temporal dimension)

Returns: Halide buffer (merged image)
'''
def merge_spatial(input):
    weight = hl.Func("raised_cosine_weights")
    output = hl.Func("merge_spatial_output")

    v, x, y = hl.Var('v'), hl.Var('x'), hl.Var('y')

    weight[v] = 0.5 - 0.5 * hl.cos(2 * math.pi * (v + 0.5) / TILE_SIZE)

    weight_00 = weight[idx_0(x)] * weight[idx_0(y)]
    weight_10 = weight[idx_1(x)] * weight[idx_0(y)]
    weight_01 = weight[idx_0(x)] * weight[idx_1(y)]
    weight_11 = weight[idx_1(x)] * weight[idx_1(y)]

    val_00 = input[idx_0(x), idx_0(y), tile_0(x), tile_0(y)]
    val_10 = input[idx_1(x), idx_0(y), tile_1(x), tile_0(y)]
    val_01 = input[idx_0(x), idx_1(y), tile_0(x), tile_1(y)]
    val_11 = input[idx_1(x), idx_1(y), tile_1(x), tile_1(y)]

    output[x, y] = hl.cast(hl.UInt(16), weight_00 * val_00
                           + weight_10 * val_10
                           + weight_01 * val_01
                           + weight_11 * val_11)

    weight.compute_root().vectorize(v, 32)

    output.compute_root().parallel(y).vectorize(x, 32)

    return output


'''
Step 2 of HDR+ pipeline: merge

images : Halide buffer
    Burst frames to be merged
alignment : Halide function
    Calculated alignment of the burst frames

Returns: Halide buffer (merged image)
'''
def merge_images(images, alignment):
    print(f'\n{"=" * 30}\nMerging images...\n{"=" * 30}')
    start = datetime.utcnow()
    merge_temporal_output = merge_temporal(images, alignment)
    merge_spatial_output = merge_spatial(merge_temporal_output)

    print(f'Merging finished in {time_diff(start)} ms.\n')
    return merge_spatial_output
