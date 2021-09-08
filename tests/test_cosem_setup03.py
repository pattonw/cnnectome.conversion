import torch
import numpy as np

from funlib.learn.torch.models import UNet

from cnnectome.conversion import transfer_weights, compare

from .datadir import datadir


def test_cosem_setup03(datadir):
    # generate some random input data
    input_shape = (216, 216, 216)
    np.random.seed()
    in_data = np.random.randn(*input_shape)

    # path to the tensorflow checkpoint
    tf_path = f"{(datadir / 'checkpoint_995000').resolve()}"
    print(tf_path)

    # build the torch unet equivalent
    torch_model = UNet(
        in_channels=1,
        num_fmaps=12,
        num_fmaps_out=72,
        fmap_inc_factor=6,
        downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
        upsample_channel_contraction=[False, True, True],
        constant_upsample=True,
        activation_on_upsample=True,
    )
    torch_model = torch.nn.Sequential(
        torch_model, torch.nn.Conv3d(72, 14, (1, 1, 1), (1, 1, 1))
    )

    # import weights from tensorflow model to torch model
    transfer_weights(tf_path, torch_model)
    # compare models. Will fail assertion if the two models
    # don't produce almost identical results
    compare(tf_path, torch_model, in_data)
