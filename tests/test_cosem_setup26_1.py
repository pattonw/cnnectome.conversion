import torch
import numpy as np

from funlib.learn.torch.models.unet import UNet, ConvPass, Upsample

from cnnectome.conversion import transfer_weights, compare

from .datadir import datadir


def test_cosem_setup26_1(datadir):
    # generate some random input data
    input_shape = (216, 216, 216)
    np.random.seed()
    in_data = np.random.randn(*input_shape)

    # path to the tensorflow checkpoint
    tf_path = f"{(datadir / 'unet_train_checkpoint_2580000').resolve()}"
    print(tf_path)

    # build the torch unet equivalent
    upsample_factor = (2, 2, 2)
    final_kernel_size = [(3,) * 3, (3,) * 3]
    final_feature_width = 12 * 6
    downsampling_factors = [(2,) * 3, (3,) * 3, (3,) * 3]

    torch_model = UNet(
        in_channels=1,
        num_fmaps=12,
        num_fmaps_out=72,
        fmap_inc_factor=6,
        downsample_factors=downsampling_factors,
        upsample_channel_contraction=[False, True, True],
        constant_upsample=True,
        activation_on_upsample=True,
    )
    torch_model = torch.nn.Sequential(
        torch_model,
        Upsample(
            upsample_factor,
            mode="nearest",
            in_channels=final_feature_width,
            out_channels=final_feature_width,
            activation="ReLU",
        ),
        ConvPass(
            final_feature_width,
            final_feature_width,
            final_kernel_size,
            activation="ReLU",
        ),
        torch.nn.Conv3d(72, 3, (1, 1, 1), (1, 1, 1)),
    )

    # import weights from tensorflow model to torch model
    transfer_weights(tf_path, torch_model)
    # compare models. Will fail assertion if the two models
    # don't produce almost identical results
    compare(tf_path, torch_model, in_data)
