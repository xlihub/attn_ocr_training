# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

import math


class MobileNetV3():
    """
    MobileNet v3, see https://arxiv.org/abs/1905.02244
    Args:
        scale (float): scaling factor for convolution groups proportion of mobilenet_v3.
        model_name (str): There are two modes, small and large.
        norm_type (str): normalization type, 'bn' and 'sync_bn' are supported.
        norm_decay (float): weight decay for normalization layer weights.
        conv_decay (float): weight decay for convolution layer weights.
        with_extra_blocks (bool): if extra blocks should be added.
        extra_block_filters (list): number of filter for each extra block.
    """

    def __init__(self,
                 scale=1.0,
                 model_name='small',
                 with_extra_blocks=False,
                 conv_decay=0.0,
                 norm_type='bn',
                 norm_decay=0.0,
                 extra_block_filters=[[256, 512], [128, 256], [128, 256],
                                      [64, 128]],
                 num_classes=None,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
                 for_seg=False,
                 output_stride=None):
        assert len(lr_mult_list) == 5, \
            "lr_mult_list length in MobileNetV3 must be 5 but got {}!!".format(
            len(lr_mult_list))
        self.scale = scale
        self.with_extra_blocks = with_extra_blocks
        self.extra_block_filters = extra_block_filters
        self.conv_decay = conv_decay
        self.norm_decay = norm_decay
        self.inplanes = 16
        self.end_points = []
        self.block_stride = 1
        self.num_classes = num_classes
        self.lr_mult_list = lr_mult_list
        self.curr_stage = 0
        self.for_seg = for_seg
        self.decode_point = None

        if self.for_seg:
            if model_name == "large":
                self.cfg = [
                    # k, exp, c,  se,     nl,  s,
                    [3, 16, 16, False, 'relu', 1],
                    [3, 64, 24, False, 'relu', 2],
                    [3, 72, 24, False, 'relu', 1],
                    [5, 72, 40, True, 'relu', 2],
                    [5, 120, 40, True, 'relu', 1],
                    [5, 120, 40, True, 'relu', 1],
                    [3, 240, 80, False, 'hard_swish', 2],
                    [3, 200, 80, False, 'hard_swish', 1],
                    [3, 184, 80, False, 'hard_swish', 1],
                    [3, 184, 80, False, 'hard_swish', 1],
                    [3, 480, 112, True, 'hard_swish', 1],
                    [3, 672, 112, True, 'hard_swish', 1],
                    # The number of channels in the last 4 stages is reduced by a
                    # factor of 2 compared to the standard implementation.
                    [5, 336, 80, True, 'hard_swish', 2],
                    [5, 480, 80, True, 'hard_swish', 1],
                    [5, 480, 80, True, 'hard_swish', 1],
                ]
                self.cls_ch_squeeze = 480
                self.cls_ch_expand = 1280
                self.lr_interval = 3
            elif model_name == "small":
                self.cfg = [
                    # k, exp, c,  se,     nl,  s,
                    [3, 16, 16, True, 'relu', 2],
                    [3, 72, 24, False, 'relu', 2],
                    [3, 88, 24, False, 'relu', 1],
                    [5, 96, 40, True, 'hard_swish', 2],
                    [5, 240, 40, True, 'hard_swish', 1],
                    [5, 240, 40, True, 'hard_swish', 1],
                    [5, 120, 48, True, 'hard_swish', 1],
                    [5, 144, 48, True, 'hard_swish', 1],
                    # The number of channels in the last 4 stages is reduced by a
                    # factor of 2 compared to the standard implementation.
                    [5, 144, 48, True, 'hard_swish', 2],
                    [5, 288, 48, True, 'hard_swish', 1],
                    [5, 288, 48, True, 'hard_swish', 1],
                ]
            else:
                raise NotImplementedError
        else:
            if model_name == "large":
                self.cfg = [
                    # kernel_size, expand, channel, se_block, act_mode, stride
                    [3, 16, 16, False, 'relu', 1],
                    [3, 64, 24, False, 'relu', 2],
                    [3, 72, 24, False, 'relu', 1],
                    [5, 72, 40, True, 'relu', 2],
                    [5, 120, 40, True, 'relu', 1],
                    [5, 120, 40, True, 'relu', 1],
                    [3, 240, 80, False, 'hard_swish', 2],
                    [3, 200, 80, False, 'hard_swish', 1],
                    [3, 184, 80, False, 'hard_swish', 1],
                    [3, 184, 80, False, 'hard_swish', 1],
                    [3, 480, 112, True, 'hard_swish', 1],
                    [3, 672, 112, True, 'hard_swish', 1],
                    [5, 672, 160, True, 'hard_swish', 2],
                    [5, 960, 160, True, 'hard_swish', 1],
                    [5, 960, 160, True, 'hard_swish', 1],
                ]
                self.cls_ch_squeeze = 960
                self.cls_ch_expand = 1280
                self.lr_interval = 3
            elif model_name == "small":
                self.cfg = [
                    # kernel_size, expand, channel, se_block, act_mode, stride
                    [3, 16, 16, True, 'relu', 2],
                    [3, 72, 24, False, 'relu', 2],
                    [3, 88, 24, False, 'relu', 1],
                    [5, 96, 40, True, 'hard_swish', 2],
                    [5, 240, 40, True, 'hard_swish', 1],
                    [5, 240, 40, True, 'hard_swish', 1],
                    [5, 120, 48, True, 'hard_swish', 1],
                    [5, 144, 48, True, 'hard_swish', 1],
                    [5, 288, 96, True, 'hard_swish', 2],
                    [5, 576, 96, True, 'hard_swish', 1],
                    [5, 576, 96, True, 'hard_swish', 1],
                ]
                self.cls_ch_squeeze = 576
                self.cls_ch_expand = 1280
                self.lr_interval = 2
            else:
                raise NotImplementedError

        if self.for_seg:
            self.modify_bottle_params(output_stride)

    def modify_bottle_params(self, output_stride=None):
        if output_stride is not None and output_stride % 2 != 0:
            raise Exception("output stride must to be even number")
        if output_stride is None:
            return
        else:
            stride = 2
            for i, _cfg in enumerate(self.cfg):
                stride = stride * _cfg[-1]
                if stride > output_stride:
                    s = 1
                    self.cfg[i][-1] = s

    def _conv_bn_layer(self,
                       input,
                       filter_size,
                       num_filters,
                       stride,
                       padding,
                       num_groups=1,
                       if_act=True,
                       act=None,
                       name=None,
                       use_cudnn=True):
        lr_idx = self.curr_stage // self.lr_interval
        lr_idx = min(lr_idx, len(self.lr_mult_list) - 1)
        lr_mult = self.lr_mult_list[lr_idx]
        if self.num_classes:
            regularizer = None
        else:
            regularizer = L2Decay(self.conv_decay)
        conv_param_attr = ParamAttr(
            name=name + '_weights',
            learning_rate=lr_mult,
            regularizer=regularizer)
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=conv_param_attr,
            bias_attr=False)
        bn_name = name + '_bn'
        bn_param_attr = ParamAttr(
            name=bn_name + "_scale", regularizer=L2Decay(self.norm_decay))
        bn_bias_attr = ParamAttr(
            name=bn_name + "_offset", regularizer=L2Decay(self.norm_decay))
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            if act == 'relu':
                bn = fluid.layers.relu(bn)
            elif act == 'hard_swish':
                bn = self._hard_swish(bn)
            elif act == 'relu6':
                bn = fluid.layers.relu6(bn)
        return bn

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _hard_swish(self, x):
        return x * fluid.layers.relu6(x + 3) / 6.

    def _se_block(self, input, num_out_filter, ratio=4, name=None):
        lr_idx = self.curr_stage // self.lr_interval
        lr_idx = min(lr_idx, len(self.lr_mult_list) - 1)
        lr_mult = self.lr_mult_list[lr_idx]

        num_mid_filter = int(num_out_filter // ratio)
        pool = fluid.layers.pool2d(
            input=input, pool_type='avg', global_pooling=True, use_cudnn=False)
        conv1 = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=num_mid_filter,
            act='relu',
            param_attr=ParamAttr(
                name=name + '_1_weights', learning_rate=lr_mult),
            bias_attr=ParamAttr(
                name=name + '_1_offset', learning_rate=lr_mult))
        conv2 = fluid.layers.conv2d(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            act='hard_sigmoid',
            param_attr=ParamAttr(
                name=name + '_2_weights', learning_rate=lr_mult),
            bias_attr=ParamAttr(
                name=name + '_2_offset', learning_rate=lr_mult))

        scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        return scale

    def _residual_unit(self,
                       input,
                       num_in_filter,
                       num_mid_filter,
                       num_out_filter,
                       stride,
                       filter_size,
                       act=None,
                       use_se=False,
                       name=None):
        input_data = input
        conv0 = self._conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_mid_filter,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + '_expand')
        if self.block_stride == 16 and stride == 2:
            self.end_points.append(conv0)
        conv1 = self._conv_bn_layer(
            input=conv0,
            filter_size=filter_size,
            num_filters=num_mid_filter,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            if_act=True,
            act=act,
            num_groups=num_mid_filter,
            use_cudnn=False,
            name=name + '_depthwise')

        if self.curr_stage == 5:
            self.decode_point = conv1

        if use_se:
            conv1 = self._se_block(
                input=conv1, num_out_filter=num_mid_filter, name=name + '_se')

        conv2 = self._conv_bn_layer(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            stride=1,
            padding=0,
            if_act=False,
            name=name + '_linear')
        if num_in_filter != num_out_filter or stride != 1:
            return conv2
        else:
            return fluid.layers.elementwise_add(
                x=input_data, y=conv2, act=None)

    def _extra_block_dw(self,
                        input,
                        num_filters1,
                        num_filters2,
                        stride,
                        name=None):
        pointwise_conv = self._conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=int(num_filters1),
            stride=1,
            padding="SAME",
            act='relu6',
            name=name + "_extra1")
        depthwise_conv = self._conv_bn_layer(
            input=pointwise_conv,
            filter_size=3,
            num_filters=int(num_filters2),
            stride=stride,
            padding="SAME",
            num_groups=int(num_filters1),
            act='relu6',
            use_cudnn=False,
            name=name + "_extra2_dw")
        normal_conv = self._conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2),
            stride=1,
            padding="SAME",
            act='relu6',
            name=name + "_extra2_sep")
        return normal_conv

    def __call__(self, input):
        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        blocks = []

        #conv1
        conv = self._conv_bn_layer(
            input,
            filter_size=3,
            num_filters=self.make_divisible(inplanes * scale),
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv1')
        i = 0
        inplanes = self.make_divisible(inplanes * scale)
        for layer_cfg in cfg:
            self.block_stride *= layer_cfg[5]
            if layer_cfg[5] == 2:
                blocks.append(conv)
            conv = self._residual_unit(
                input=conv,
                num_in_filter=inplanes,
                num_mid_filter=self.make_divisible(scale * layer_cfg[1]),
                num_out_filter=self.make_divisible(scale * layer_cfg[2]),
                act=layer_cfg[4],
                stride=layer_cfg[5],
                filter_size=layer_cfg[0],
                use_se=layer_cfg[3],
                name='conv' + str(i + 2))
            inplanes = self.make_divisible(scale * layer_cfg[2])
            i += 1
            self.curr_stage = i
        blocks.append(conv)

        if self.for_seg:
            conv = self._conv_bn_layer(
                input=conv,
                filter_size=1,
                num_filters=self.make_divisible(scale * self.cls_ch_squeeze),
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act='hard_swish',
                name='conv_last')

            return conv, self.decode_point

        if self.num_classes:
            conv = self._conv_bn_layer(
                input=conv,
                filter_size=1,
                num_filters=int(scale * self.cls_ch_squeeze),
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act='hard_swish',
                name='conv_last')

            conv = fluid.layers.pool2d(
                input=conv,
                pool_type='avg',
                global_pooling=True,
                use_cudnn=False)
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=self.cls_ch_expand,
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(name='last_1x1_conv_weights'),
                bias_attr=False)
            conv = self._hard_swish(conv)
            drop = fluid.layers.dropout(x=conv, dropout_prob=0.2)
            out = fluid.layers.fc(input=drop,
                                  size=self.num_classes,
                                  param_attr=ParamAttr(name='fc_weights'),
                                  bias_attr=ParamAttr(name='fc_offset'))
            return out

        if not self.with_extra_blocks:
            return blocks

        # extra block
        conv_extra = self._conv_bn_layer(
            conv,
            filter_size=1,
            num_filters=int(scale * cfg[-1][1]),
            stride=1,
            padding="SAME",
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv' + str(i + 2))
        self.end_points.append(conv_extra)
        i += 1
        for block_filter in self.extra_block_filters:
            conv_extra = self._extra_block_dw(conv_extra, block_filter[0],
                                              block_filter[1], 2,
                                              'conv' + str(i + 2))
            self.end_points.append(conv_extra)
            i += 1

        return self.end_points
