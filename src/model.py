import torch as pt


class SAL(pt.nn.Module):

    def __init__(self, conv_channels=[4, 8, 16, 32, 64, 1024], linear_features=[512, 512, 256, 128]):
        super().__init__()

        conv_layers = []
        conv_in_out = zip(conv_channels, conv_channels[1:])
        for in_channels, out_channels in conv_in_out:
            conv_layers += [
                    pt.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    pt.ReLU(),
                       ]
        self.bridge = 
        self.conv_block = pt.nn.Sequential(conv_layers)
        
        
        linear_layers = []
        linear_in_out = zip([conv_channels[0], *linear_features], linear_features[1:])
        for in_features, out_features in linear_in_out:
            linear_layers += [
                    pt.nn.Linear(in_features=in_features, out_features=out_features),
                    pt.ReLU(),
                       ]
        self.linear_block = pt.nn.Sequential(linear_layers)

    def forward(self, env_map, speed):
        env_embed = self.conv_block(env_map).flatten(-2)
        path = self.linear_block(pt.cat((env_embed, speed), dim=-1))

        return env_embed, path


if __name__ == "__main__":
    sal = SAL()
    env_map = pt.randn(4, 256, 256)
