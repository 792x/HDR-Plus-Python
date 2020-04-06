from datetime import datetime
import halide as hl

# Global constants
TILE_SIZE = 32
TILE_SIZE_2 = 16
MINIMUM_OFFSET = -168
MAXIMUM_OFFSET = 126
DOWNSAMPLE_RATE = 4
DENOISE_PASSES = 1
SHARPEN_STRENGTH = 2
TONE_MAP_PASSES = 4

'''
Get the difference between a start and end time (or current time if none given) in ms

start : datetime
    Start time
end : datetime
    End time

Returns: int
'''


def time_diff(start, end=None):
    if not end:
        end = datetime.utcnow()
    return int((end - start).total_seconds() * 1000)


"""
Point object which stores the coordinates x and y

x : float
    x-coordinate
y : float
    y-coordinate
"""


class Point:
    def __init__(self, x=None, y=None):
        if x is None and y is None:
            self.x = hl.cast(hl.Int(16), 0)
            self.y = hl.cast(hl.Int(16), 0)
        elif x is not None and y is None:
            if type(x) is hl.FuncRef:
                hl.Tuple(x)
                self.x = hl.cast(hl.Int(16), x[0])
                self.y = hl.cast(hl.Int(16), x[1])
            elif type(x) is tuple:
                self.x = hl.cast(hl.Int(16), x[0])
                self.y = hl.cast(hl.Int(16), x[1])
        else:
            self.x = hl.cast(hl.Int(16), x)
            self.y = hl.cast(hl.Int(16), y)

    def get_tuple(self):
        return self.x, self.y

    def clamp(self, min_p, max_p):
        return Point(hl.clamp(self.x, min_p.x, max_p.x), hl.clamp(self.y, min_p.y, max_p.y))

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return (self.x, self.y)[idx]

    # Point addition
    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    # Point subtraction
    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    # Scalar multiplication
    def __mul__(self, n: int):
        return Point(self.x * n, self.y * n)

    # Scalar multiplication with self on the right-hand side
    def __rmul__(self, n: int):
        return Point(self.x * n, self.y * n)

    # Point negation
    def __neg__(self):
        return Point(-self.x, -self.y)


def gaussian_down4(input, name):
    output = hl.Func(name)
    k = hl.Func(name + "_filter")
    x, y, n = hl.Var("x"), hl.Var("y"), hl.Var('n')
    rdom = hl.RDom([(-2, 5), (-2, 5)])

    k[x, y] = 0
    k[-2, -2] = 2
    k[-1, -2] = 4
    k[0, -2] = 5
    k[1, -2] = 4
    k[2, -2] = 2
    k[-2, -1] = 4
    k[-1, -1] = 9
    k[0, -1] = 12
    k[1, -1] = 9
    k[2, -1] = 4
    k[-2, 0] = 5
    k[-1, 0] = 12
    k[0, 0] = 15
    k[1, 0] = 12
    k[2, 0] = 5
    k[-2, 1] = 4
    k[-1, 1] = 9
    k[0, 1] = 12
    k[1, 1] = 9
    k[2, 1] = 4
    k[-2, 2] = 2
    k[-1, 2] = 4
    k[0, 2] = 5
    k[1, 2] = 4
    k[2, 2] = 2

    output[x, y, n] = hl.cast(hl.UInt(16),
                              hl.sum(hl.cast(hl.UInt(32), input[4 * x + rdom.x, 4 * y + rdom.y, n] * k[rdom.x, rdom.y]))
                              / 159)

    k.compute_root().parallel(y).parallel(x)
    output.compute_root().parallel(y).vectorize(x, 16)

    return output


def box_down2(input, name):
    output = hl.Func(name)

    x, y, n = hl.Var("x"), hl.Var("y"), hl.Var('n')
    rdom = hl.RDom([(0, 2), (0, 2)])

    output[x, y, n] = hl.cast(hl.UInt(16), hl.sum(hl.cast(hl.UInt(32), input[2 * x + rdom.x, 2 * y + rdom.y, n])) / 4)

    output.compute_root().parallel(y).vectorize(x, 16)

    return output


def prev_tile(t):
    return (t - 1) / DOWNSAMPLE_RATE


def idx_layer(t, i):
    return t * TILE_SIZE_2 / 2 + i


def idx_im(t, i):
    return t * TILE_SIZE_2 + i


def idx_0(e):
    return e % TILE_SIZE_2 + TILE_SIZE_2


def idx_1(e):
    return e % TILE_SIZE_2


def tile_0(e):
    return e / TILE_SIZE_2 - 1


def tile_1(e):
    return e / TILE_SIZE_2
