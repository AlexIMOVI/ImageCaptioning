import torch
import json


class utils:
    __GLOBAL_STATS__ = {}

    @staticmethod
    def setup_gpus(gpu, use_cudnn):
        dtype = torch.FloatTensor
        actual_use_cudnn = False
        if gpu >= 0:
            import torch.cuda
            import torch.backends.cudnn as cudnn
            torch.cuda.set_device(gpu)
            dtype = torch.cuda.FloatTensor
            if use_cudnn == 1:
                cudnn.benchmark = True
                actual_use_cudnn = True
        return dtype, actual_use_cudnn

    @staticmethod
    def build_loss_string(losses):
        x = ''
        for k, v in losses.items():
            if k != 'total_loss':
                x += f'{k}: {v:.5f}, '
        x += f' [total: {losses["total_loss"]:.3f}]'
        return x

    @staticmethod
    def build_timing_string(timings):
        x = ''
        for k, v in timings.items():
            x += f'timing {k}: {v * 1000:.3f}ms\n'
        return x

    @staticmethod
    def getopt(opt, key, default_value=None):
        if default_value is None and (opt is None or key not in opt):
            raise ValueError(f'Error: required key {key} was not provided in an opt.')
        return opt[key] if opt and key in opt else default_value

    @staticmethod
    def ensureopt(opt, key):
        if opt is None or key not in opt:
            raise ValueError(f'Error: required key {key} was not provided.')

    @staticmethod
    def read_json(path):
        with open(path, 'r') as file:
            return json.load(file)

    @staticmethod
    def write_json(path, j):
        with open(path, 'w') as file:
            json.dump(j, file)

    @staticmethod
    def dict_average(dicts):
        dict_sum = {}
        n = len(dicts)
        for d in dicts:
            for k, v in d.items():
                dict_sum[k] = dict_sum.get(k, 0) + v.item()
        return {k: v / n for k, v in dict_sum.items()}

    @staticmethod
    def count_keys(t):
        return len(t)

    @staticmethod
    def average_values(t):
        n = len(t)
        v_sum = sum(t.values())
        return v_sum / n
