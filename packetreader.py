import re


def isPositive(integerx):
    return True if integerx > 0 else False


def get_time(stringx):
    unadjusted_time = re.search(r'\d+:\d+:\d+\.\d+\s', stringx).group()[:-1]
    adjusted_time = re.split(r'\.|:', unadjusted_time)
    adjusted_time[3] = "0." + adjusted_time[3]
    return float(adjusted_time[0]) * 3600 + float(adjusted_time[1]) * 60 + float(adjusted_time[2]) + float(
        adjusted_time[3])


def check_push_flag(stringx):
    return False if not re.search(r'Flags\s\[P\.\]', stringx) else True


def check_if_send(stringx, stringy):
    return True if (get_source(stringx) == stringy) else False


def get_source(stringx):
    return re.search(r'IP \d+\.\d+\.\d+\.\d+\.', stringx).group()[3:-1]


def get_destination(stringx):
    return re.search(r'>\s\d+\.\d+\.\d+\.\d+\.', stringx).group()[2:-1]
