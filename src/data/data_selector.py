from .data_utils import load_data


def select_data(args, train=True):
    train_data, val_data, test_data = load_data(args.data_name, args.data_dir_path)
    if not train:
        return test_data
    return val_data, train_data

