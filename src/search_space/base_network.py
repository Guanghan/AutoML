"""
@author: Guanghan Ning
@file: base_network.py
@time: 10/7/20 12:11 上午
@file_desc: base class for a differentiable super network;
            large proportion of code borrowed from:
            https://github.com/huawei-noah/vega/blob/master/vega/search_space/networks/pytorch/network.py
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class Network(nn.Module):
    """Base class for differentiable super networks."""

    def __init__(self, network_seqs=None, is_freeze=False, condition=None,
                 out_list=None, **kwargs):
        super(Network, self).__init__()
        self.is_freeze = is_freeze
        self.condition = condition
        self.out_list = out_list
        if network_seqs is None:
            return
        for index, seq in enumerate(network_seqs):
            if isinstance(seq, list):
                model = self.add_model(str(index), seq)
            else:
                model = seq
            self.add_module(str(index), model)

    def add_model(self, name, seq):
        model = nn.Sequential()
        if not isinstance(seq, list):
            model.add_module(name, seq)
        else:
            for index, item in enumerate(seq):
                model.add_module(name + str(index), item)
        return model

    @property
    def input_shape(self):
        """Get the model input tensor shape."""
        raise NotImplementedError

    @property
    def output_shape(self):
        """Get the model output tensor shape."""
        raise NotImplementedError

    @property
    def model_layers(self):
        """Get the model layers."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        """Build a model from config."""
        raise NotImplementedError

    def train(self, mode=True):
        """Train setting."""
        super().train(mode)
        if self.is_freeze and mode:
            # freeze BatchNorm for now
            for m in self.modules():
                if isinstance(m, nn.batchnorm._BatchNorm):
                    self._freeze(m)

    def _freeze(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, *args, forward_train=True, **kwargs):
        """Call default forward function."""
        if len(args) == 1:
            if forward_train:
                return self.forward_train(*args, **kwargs)
            else:
                return self.forward_valid(*args, **kwargs)
        else:
            if forward_train:
                return self.multi_forward_train(*args, **kwargs)
            else:
                return self.multi_forward_valid(*args, **kwargs)

    def forward_train(self, input, **kwargs):
        """Call train forward function."""
        models = self.children()
        if self.condition == 'add':
            # for add
            output = None
            for model in models:
                if output is None:
                    output = model(input)
                else:
                    output = output + model(input)
        elif self.condition == 'concat':
            # for merge
            outputs = []
            for model in models:
                outputs.append(model(input))
            output = torch.cat(outputs, 1)
        elif self.condition == 'interpolate':
            output = input
            for model in models:
                output = model(output)
            output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)
        elif self.condition == 'merge':
            output = self.merge(input, models)
        else:
            # for seq
            output = self.seq_forward(input, models, **kwargs)
        return output

    def merge(self, input, models):
        """Merge all models."""
        output = input
        outputs = []
        for model in models:
            model_out = model(output)
            if isinstance(model_out, tuple):
                model_out = list(model_out)
                outputs.extend(model_out)
            else:
                outputs.append(model_out)
            output = tuple(outputs)
        return output

    def seq_forward(self, input, models, **kwargs):
        """Call sequential train forward function."""
        output = input
        if self.out_list is None:
            for model in models:
                output = model(output)
        else:
            outputs = []
            models = list(models)
            for idx, model in enumerate(models):
                output = model(output)
                if idx in self.out_list:
                    outputs.append(output)
            output = outputs
        return output

    def Input_list_forward(self, input, models, **kwargs):
        """Call list of input train forward function."""
        if self.out_list is None:
            if isinstance(input, list):
                outputs = []
                for model, idx in zip(models, [i for i in range(len(input))]):
                    output = model(input[idx])
                    outputs.append(output)
                output = outputs
            else:
                raise ValueError("Input must list!")
        else:
            input = list(input)
            for model, idx in zip(models, self.out_list):
                if isinstance(idx, list):
                    assert len(idx) == 2
                    output = model(input[idx[0]], input[idx[1]])
                    input.append(output)
                else:
                    input.append(model(input[idx]))
            output = input
        return output

    def forward_valid(self, input, **kwargs):
        """Call test forward function."""
        raise NotImplementedError

    def mutil_forward_train(self, *args, **kwargs):
        """Call mutil input train forward function."""
        models = list(self.children())
        output = []
        for idx in range(len(args)):
            output.append(models[idx](args[idx]))
        output = models[-1](*tuple(output))
        return output

    def multi_forward_valid(self, input, **kwargs):
        """Call test forward function."""
        raise NotImplementedError
