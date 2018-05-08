"""
This module contains function to convert dates in appropriate format
"""

from datetime import datetime, timedelta

def matlab2ymdhms(time):
    """
    Convert Matlab format to year/month/dat/hour/minute/second

    Input:
        type time =  float
        time = Number of days since January 1, 0000 (Matlab format)
    Output:
        type output = tuple of 6 integers
        output = year, month, day, hour, minute, second
    """    
    myday = datetime.fromordinal(int(time)) + \
        timedelta(days=time % 1) - timedelta(days=366)
    year = myday.year
    month = myday.month
    day = myday.day
    hour = myday.hour
    minute = myday.minute
    second = myday.second
    microsecond = myday.microsecond
    rsecond = int(round(second + microsecond / 1000000.0))
    if (rsecond == 60):
        minute = minute + 1
        rsecond = 0
    if (minute == 60):
        hour = hour + 1
        minute = 0
    if (hour == 24):
        day = day + 1
        hour = 0
    return (year, month, day, hour, minute, rsecond)
    