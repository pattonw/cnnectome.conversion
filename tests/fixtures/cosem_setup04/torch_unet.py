import torch
from funlib.learn.torch.models.unet import UNet, Upsample, ConvPass
from funlib.geometry import Coordinate

# voxel size parameters
voxel_size_labels = Coordinate((2,) * 3)
voxel_size = Coordinate((4,) * 3)
voxel_size_input = Coordinate((8,) * 3)

# network parameters copied from unet_template.py
padding = "valid"
constant_upsample = True
trans_equivariant = True
feature_widths_down = [12, 12 * 6, 12 * 6 ** 2, 12 * 6 ** 3]
feature_widths_up = [12 * 6, 12 * 6, 12 * 6 ** 2, 12 * 6 ** 3]
downsampling_factors = [(2,) * 3, (3,) * 3, (3,) * 3]
kernel_sizes_down = [
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
]
kernel_sizes_up = [[(3,) * 3, (3,) * 3], [(3,) * 3, (3,) * 3], [(3,) * 3, (3,) * 3]]

# additional network parameters for upsampling network
upsample_factor = tuple(voxel_size_input / voxel_size)
final_kernel_size = [(3,) * 3, (3,) * 3]
final_feature_width = 12 * 6


# build the torch unet equivalent
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
        mode="constant",
        in_channels=final_feature_width,
        out_channels=final_feature_width,
        activation="ReLU",
    ),
    ConvPass(
        final_feature_width, final_feature_width, final_kernel_size, activation="ReLU"
    ),
    torch.nn.Conv3d(72, 14, (1, 1, 1), (1, 1, 1)),
)