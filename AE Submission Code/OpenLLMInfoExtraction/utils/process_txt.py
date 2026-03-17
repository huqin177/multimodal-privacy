import chardet

def det_encoding(path):
    assert ('.txt' in path)
    with open(path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
    return result

def open_txt(path, data_num='all'):
    try:
        with open(path, encoding='utf-8', errors='ignore') as file:
            lines = [line.rstrip() for line in file]
    except:
        raise ValueError(f"{path}")
    if data_num == 'all':
        return lines
    else:
        try:
            return lines[:int(data_num)]
        except ValueError:
            raise ValueError(f'ERROR: {data_num} is not a valid data_num')