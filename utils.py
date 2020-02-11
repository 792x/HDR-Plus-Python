from datetime import datetime, timedelta


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