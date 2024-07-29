import mne

def read_bdf_data(file_name):
    # 加载BDF文件
    mne.set_log_level('ERROR')
    raw = mne.io.read_raw_bdf(file_name, preload=True)

    # 创建EEG数据结构
    EEG = {}
    # 使用注释创建事件
    target = "ssvep01sub01"
    if target in file_name:
        event_id = {'1': 10, '8': 1, '9': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8}
    else:
        event_id = {'10': 10, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}

    events, _ = mne.events_from_annotations(raw, event_id=event_id)

    EEG['event'] = events.tolist()
    EEG['data'] = raw.get_data()

    return EEG


# 用法示例
if __name__ == '__main__':
    filename = 'D:/Document/Desktop/SRH/11.ssvep/data/mdr/wr_20240407_mdr_8/mdr/mdr.bdf'
    EEG = read_bdf_data(filename)
    print(EEG)


