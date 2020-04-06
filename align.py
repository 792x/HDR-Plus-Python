import math

from datetime import datetime
import halide as hl

from utils import time_diff, Point, gaussian_down4, box_down2, prev_tile, idx_layer, TILE_SIZE_2, DOWNSAMPLE_RATE

'''
Determines the best offset for tiles of the image at a given resolution, 
provided the offsets for the layer above

layer : numpy.ndarray
    The layer for which the offset needs to be calculated
prev_alignment : Point
    Alignment of the previous layer
prev_min : Point
    Min search region
prev_max : Point
    Max search region

Returns: Point
'''


def align_layer(layer, prev_alignment, prev_min, prev_max):
    scores = hl.Func(layer.name() + "_scores")
    alignment = hl.Func(layer.name() + "_alignment")
    xi, yi, tx, ty, n = hl.Var("xi"), hl.Var("yi"), hl.Var('tx'), hl.Var('ty'), hl.Var('n')
    rdom0 = hl.RDom([(0, 16), (0, 16)])
    rdom1 = hl.RDom([(-4, 8), (-4, 8)])

    prev_offset = DOWNSAMPLE_RATE * Point(prev_alignment[prev_tile(tx), prev_tile(ty), n]).clamp(prev_min, prev_max)

    x0 = idx_layer(tx, rdom0.x)
    y0 = idx_layer(ty, rdom0.y)
    x = x0 + prev_offset.x + xi
    y = y0 + prev_offset.y + yi

    ref_val = layer[x0, y0, 0]
    alt_val = layer[x, y, n]

    dist = hl.abs(hl.cast(hl.Int(32), ref_val) - hl.cast(hl.Int(32), alt_val))

    scores[xi, yi, tx, ty, n] = hl.sum(dist)

    alignment[tx, ty, n] = Point(hl.argmin(scores[rdom1.x, rdom1.y, tx, ty, n])) + prev_offset

    scores.compute_at(alignment, tx).vectorize(xi, 8)

    alignment.compute_root().parallel(ty).vectorize(tx, 16)

    return alignment


'''
Step 1 of HDR+ pipeline: align

images : list of numpy ndarray
    The raw burst images
grayscale : list of numpy ndarray
    Grayscale versions of the images

Returns: list of numpy ndarray (aligned images)
'''


def align_images(images):
    print(f'\n{"=" * 30}\nAligning images...\n{"=" * 30}')
    start = datetime.utcnow()

    alignment_3 = hl.Func("layer_3_alignment")
    alignment = hl.Func("alignment")

    tx, ty, n = hl.Var('tx'), hl.Var('ty'), hl.Var('n')

    print('Subsampling image layers...')
    imgs_mirror = hl.BoundaryConditions.mirror_interior(images, [(0, images.width()), (0, images.height())])
    layer_0 = box_down2(imgs_mirror, "layer_0")
    layer_1 = gaussian_down4(layer_0, "layer_1")
    layer_2 = gaussian_down4(layer_1, "layer_2")

    min_search = Point(-4, -4)
    max_search = Point(3, 3)

    min_3 = Point(0, 0)
    min_2 = DOWNSAMPLE_RATE * min_3 + min_search
    min_1 = DOWNSAMPLE_RATE * min_2 + min_search

    max_3 = Point(0, 0)
    max_2 = DOWNSAMPLE_RATE * max_3 + max_search
    max_1 = DOWNSAMPLE_RATE * max_2 + max_search

    print('Aligning layers...')
    alignment_3[tx, ty, n] = Point(0, 0)

    alignment_2 = align_layer(layer_2, alignment_3, min_3, max_3)
    alignment_1 = align_layer(layer_1, alignment_2, min_2, max_2)
    alignment_0 = align_layer(layer_0, alignment_1, min_1, max_1)

    num_tx = math.floor(images.width() / TILE_SIZE_2 - 1)
    num_ty = math.floor(images.height() / TILE_SIZE_2 - 1)

    alignment[tx, ty, n] = 2 * Point(alignment_0[tx, ty, n])

    alignment_repeat = hl.BoundaryConditions.repeat_edge(alignment, [(0, num_tx), (0, num_ty)])

    print(f'Alignment finished in {time_diff(start)} ms.\n')
    return alignment_repeat
