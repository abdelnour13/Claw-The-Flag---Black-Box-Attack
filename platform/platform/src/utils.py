def parse_file_size(size : str) -> int:

    units = {
        'k' : 1024,
        'm' : 1024 ** 2,
        'g' : 1024 ** 3
    }

    value, unit = size[:-1], size[-1]

    return int(value) * units[unit]