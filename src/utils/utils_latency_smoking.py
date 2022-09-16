"""
@author: Guanghan Ning
@file: utils_latency.py
@time: 11/26/20 8:18 下午
@file_desc: Utility to create and load Lookup Table (Lut)
"""
import torch
import time
import pickle
from collections import OrderedDict
from src.search_space.darts_blocks import none, avg_pool_3x3, max_pool_3x3, skip_connect, \
	 sep_conv_3x3, sep_conv_5x5, dil_conv_3x3, dil_conv_5x5, PreOneStem


LAT_TIMES  = 1000
INIT_TIMES = 100


PRIMITIVES = [
    'none',
    'avg_pool_3x3',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
	'PreOneStem',
]


OPS = {
    'none': lambda  channel_in, channel_out, kernel_size, stride, padding, dilation: none(convert_desc_based_on_op('none', channel_in, channel_out, kernel_size, stride, padding, dilation)),
    'max_pool_3x3': lambda channel_in, channel_out, kernel_size, stride, padding, dilation: max_pool_3x3(convert_desc_based_on_op('max_pool_3x3', channel_in, channel_out, kernel_size, stride, padding, dilation)),
    'avg_pool_3x3': lambda channel_in, channel_out, kernel_size, stride, padding, dilation: avg_pool_3x3(convert_desc_based_on_op('avg_pool_3x3', channel_in, channel_out, kernel_size, stride, padding, dilation)),
    'skip_connect': lambda channel_in, channel_out, kernel_size, stride, padding, dilation: skip_connect(convert_desc_based_on_op('skip_connect', channel_in, channel_out, kernel_size, stride, padding, dilation)),
    'sep_conv_3x3': lambda channel_in, channel_out, kernel_size, stride, padding, dilation: sep_conv_3x3(convert_desc_based_on_op('sep_conv_3x3', channel_in, channel_out, kernel_size, stride, padding, dilation)),
    'sep_conv_5x5': lambda channel_in, channel_out, kernel_size, stride, padding, dilation: sep_conv_5x5(convert_desc_based_on_op('sep_conv_5x5', channel_in, channel_out, kernel_size, stride, padding, dilation)),
    'dil_conv_3x3': lambda channel_in, channel_out, kernel_size, stride, padding, dilation: dil_conv_3x3(convert_desc_based_on_op('dil_conv_3x3', channel_in, channel_out, kernel_size, stride, padding, dilation)),
    'dil_conv_5x5': lambda channel_in, channel_out, kernel_size, stride, padding, dilation: dil_conv_5x5(convert_desc_based_on_op('dil_conv_5x5', channel_in, channel_out, kernel_size, stride, padding, dilation)),
	'PreOneStem': lambda channel_in, channel_out, kernel_size, stride, padding, dilation: PreOneStem(convert_desc_based_on_op('PreOneStem', channel_in, channel_out, kernel_size, stride, padding, dilation)),
}


def convert_desc_based_on_op(op_name, channel_in, channel_out, kernel_size, stride, padding, dilation):
	desc = {}
	assert(channel_in == channel_out)  # darts specific
	desc['C'] = channel_in
	desc['kernel_size'] = kernel_size
	desc['stride'] = stride
	desc['padding'] = padding
	desc['dilation'] = dilation
	if op_name == 'PreOneStem':
		desc['stem_multi'] = 3
	return desc


class AverageMeter(object):
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def measure_latency_in_ms(model, input_shape, is_cuda):
	lat = AverageMeter()
	model.eval()

	x = torch.randn(input_shape)
	if is_cuda:
		model = model.cuda()
		x = x.cuda()
	else:
		model = model.cpu()
		x = x.cpu()

	with torch.no_grad():
		for _ in range(INIT_TIMES):
			output = model(x)

		for _ in range(LAT_TIMES):
			tic = time.time()
			output = model(x)
			toc = time.time()
			lat.update(toc-tic, x.size(0))

	return lat.avg * 1000 # save as ms


def get_latency_lut(is_cuda = True):
	latency_lut = OrderedDict()

	input_size_list = [1, 2, 4, 8, 16, 32, 64, 128]
	channel_list = [2, 4, 8, 16, 32, 64, 96, 128, 224, 256]
	stride_list = [1, 2]
	for idx in range(len(PRIMITIVES)):
		block_name = PRIMITIVES[idx]

		if block_name[-1] == '3':
			kernel_size = 3
		elif block_name[-1] == '5':
			kernel_size = 5
		else:
			kernel_size = None

		for input_size in input_size_list:
			for channel_in in channel_list:
				for stride in stride_list:
					if stride == 2 and input_size == 1: continue
					if block_name == "PreOneStem":
						channel_in = 3

					block = OPS[block_name](channel_in, channel_in, kernel_size, stride, padding=None, dilation=None)
					shape = (32, channel_in, input_size, input_size) if is_cuda else (1, channel_in, input_size, input_size)
					key = '{}_{}x{}_c{}_k{}_s{}'.format(block_name, input_size, input_size, channel_in, kernel_size, stride)
					latency = measure_latency_in_ms(block, shape, is_cuda)
					latency_lut[key] = latency
					print("{}: res={}, channel={}, stride={}, avg_latency:{}".format(block_name, input_size, channel_in, stride, latency))
	return latency_lut



if __name__ == '__main__':
	print('measure latency on gpu......')
	latency_lookup = get_latency_lut(is_cuda=True)
	with open('latency_gpu_darts_smoking.pkl', 'wb') as f:
		pickle.dump(latency_lookup, f)

	# print('measure latency on cpu......')
	# latency_lookup = get_latency_lut(is_cuda=False)
	# with open('latency_cpu_darts_smoking.pkl', 'wb') as f:
	# 	pickle.dump(latency_lookup, f)
