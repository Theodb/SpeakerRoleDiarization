def to_timestamp(t: int, separator=',') -> str:
    """
    376 -> 00:00:03,760
    1344 -> 00:00:13,440

    Implementation from `whisper.cpp/examples/main`

    :param t: input time from whisper timestamps
    :param separator: seprator between seconds and milliseconds
    :return: time representation in hh: mm: ss[separator]ms
    """
    # logic exactly from whisper.cpp

    msec = t * 10
    hr = msec // (1000 * 60 * 60)
    msec = msec - hr * (1000 * 60 * 60)
    min = msec // (1000 * 60)
    msec = msec - min * (1000 * 60)
    sec = msec // 1000
    msec = msec - sec * 1000
    return f"{int(hr):02,.0f}:{int(min):02,.0f}:{int(sec):02,.0f}{separator}{int(msec):03,.0f}"

def to_seconds(t: int) -> float:
    """
    Convert a given timestamp (in Whisper's format) into seconds.

    Implementation from `whisper.cpp/examples/main`

    :param t: input time from whisper timestamps
    :return: time in seconds
    """
    # Convert the input timestamp to milliseconds
    msec = t * 10
    
    # Convert milliseconds to seconds
    seconds = msec / 1000.0
    return seconds