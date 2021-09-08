import torch
import tensorflow as tf
import numpy as np


def transfer_weights(tf_path, torch_model):
    torch_unet = torch_model[0]
    num_down_kernels = [len(kernel) for kernel in torch_unet.kernel_size_down]
    num_up_kernels = [len(kernel) for kernel in torch_unet.kernel_size_up]
    # tensorflow to torch map for tensor naming convention
    tensors = {"bias": "bias", "kernel": "weight"}

    # generate list of tuple pairs: (tensorflow layer name, torch layer path)
    tf_tensor_names = []
    tf_tensor_names += [
        (
            f"unet_layer_{layer}_left_{conv}/{tensor}",
            ["0", "l_conv", f"{layer}", "conv_pass", f"{conv * 2}", tensors[tensor]],
        )
        for layer in range(len(num_down_kernels))
        for conv in range(num_down_kernels[layer])
        for tensor in ["bias", "kernel"]
    ]
    tf_tensor_names += [
        (
            f"unet_layer_{layer}_right_{conv}/{tensor}",
            [
                "0",
                "r_conv",
                "0",
                f"{layer}",
                "conv_pass",
                f"{conv * 2}",
                tensors[tensor],
            ],
        )
        for layer in range(len(num_up_kernels))
        for conv in range(num_up_kernels[layer])
        for tensor in ["bias", "kernel"]
    ]
    tf_tensor_names += [
        (f"conv_pass_0/{tensor}", ["1", tensors[tensor]])
        for tensor in ["bias", "kernel"]
    ]
    # constant upsample
    tf_tensor_names += [
        (
            f"unet_up_{layer+1}_to_{layer}_kernel_variables",
            ["0", "r_up", "0", f"{layer}", "up", "1", "weight"],
        )
        for layer in range(len(num_up_kernels))
    ]

    # Retrieve weights from TF checkpoint
    for tf_name, torch_scopes in tf_tensor_names:
        array = tf.train.load_variable(tf_path, tf_name)

        # Initiate the pointer from the main model class
        pointer = torch_model

        # We iterate along the scopes and move our pointer accordingly
        for scope in torch_scopes:
            if tf_name.endswith("kernel_variables") and scope == "weight":
                bias = getattr(pointer, "bias")
            pointer = getattr(pointer, scope)

        # handle special cases
        if tf_name.endswith("kernel_variables"):
            # weights for upsample are stored in an array and reshaped and duplicated
            # to ensure a constant upsample in tensorflow. In torch we simply duplicate
            # the data and then apply a 1x1x1 convolution. The effect is the same so
            # we just need to reshape the array into a kernel
            array = np.reshape(array, pointer.shape)
            # make sure to set bias equal to 0. Since the kernel is constructed, it seems
            # there is no bias.
            bias.data = torch.from_numpy(np.zeros([pointer.shape[0]])).float()
        elif torch_scopes[-1] == "weight":
            # tensorflow weight are stored x,y,z,out_channels,in_channels
            # torch weights are stored in_channels,out_channels,x,y,z
            transpose_dims = [-1, -2] + [x for x in range(len(array.shape) - 2)]
            array = np.transpose(array, transpose_dims)

        try:
            assert (
                pointer.shape == array.shape
            )  # Catch error if the array shapes are not identical
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise

        pointer.data = torch.from_numpy(array)

    return torch_model