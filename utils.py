from datetime import datetime, timedelta
import halide as hl

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
    return int((end - start).total_seconds()*1000)


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
