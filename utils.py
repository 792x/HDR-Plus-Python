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
        print('making new Point')
        print(x, y)
        if x is None and y is None:
            self.x = hl.cast(hl.UInt(16), 0)
            self.y = hl.cast(hl.UInt(16), 0)
        if x is not None and y is None:
            self.x = hl.cast(hl.UInt(16), x[0])
            self.y = hl.cast(hl.UInt(16), x[1])
        else:
            self.x = hl.cast(hl.UInt(16), x)
            self.y = hl.cast(hl.UInt(16), y)
        print('made new point')
        print(x, y)

    def get_tuple(self):
        return self.x, self.y

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __repr__(self):
        return f'({self.x}, {self.y})'

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
