from typing import List, Optional

import torch
import torch.nn as nn


class MNISTAutoencoder(nn.Module):
    def __init__(self, latent_space_size: int, number_of_layers: int, latent_space_activation_function: Optional[nn.Module]):
        super().__init__()

        self.input_size = 28 * 28
        self.latent_space_size = latent_space_size
        self.number_of_layers = number_of_layers

        assert self.input_size > self.latent_space_size

        # Sometimes we are unable to nicely divide the amount of compression we have to do
        # between layers. We are left with a nasty decimal reminder.
        # For example:
        # latent_space_size = 6
        # number_of_layers = 4
        # (784 - 6) / 4 = 194,5
        # We would need to compress the output by 194,5 on each layer. We can't have half of a neuron can we?
        # In this loop we are trying to accumulate this reminder and distribute it weever possible.
        # This way the amount of compression we do per-layer is ~constant.

        reduce_by_n_features_per_layer = (self.input_size - self.latent_space_size) // self.number_of_layers
        reduce_by_n_features_per_layer_reminder = (self.input_size - self.latent_space_size) / self.number_of_layers - reduce_by_n_features_per_layer

        previous_layer_out_features = self.input_size
        encoder_layers_out_features = []
        reminder = 0
        for _ in range(number_of_layers - 1):
            reduce_by_n = reduce_by_n_features_per_layer
            reminder += reduce_by_n_features_per_layer_reminder
            if reminder >= 1:
                reminder -= 1
                reduce_by_n += 1

            layer_out_features = previous_layer_out_features - reduce_by_n

            previous_layer_out_features = layer_out_features
            encoder_layers_out_features.append(layer_out_features)

        # Regardless of the reminder logic, we want to have our latent space size set
        # to what user intended.
        encoder_layers_out_features.append(self.latent_space_size)

        encoder_modules: List[nn.Module] = [
            nn.Flatten(),
        ]

        previous_layer_out_features = self.input_size
        for layer_out_features in encoder_layers_out_features:
            encoder_modules.append(nn.Linear(previous_layer_out_features, layer_out_features))
            previous_layer_out_features = layer_out_features

            is_last_layer = layer_out_features == self.latent_space_size
            if not is_last_layer:
                encoder_modules.append(nn.ReLU())

        if latent_space_activation_function is not None:
            encoder_modules.append(latent_space_activation_function)

        decoder_modules: List[nn.Module] = []

        decoder_layers_out_features = list(reversed(encoder_layers_out_features))
        decoder_layers_out_features = decoder_layers_out_features[1:]
        decoder_layers_out_features.append(self.input_size)

        previous_layer_out_features = self.latent_space_size
        for layer_out_features in decoder_layers_out_features:
            decoder_modules.append(nn.Linear(previous_layer_out_features, layer_out_features))
            previous_layer_out_features = layer_out_features

            is_last_layer = layer_out_features == self.input_size
            if not is_last_layer:
                decoder_modules.append(nn.ReLU())

        decoder_modules.append(nn.Unflatten(-1, (1, 28, 28)))

        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)

    def __str__(self):
        return f"MNISTAutoencoder:\n    Latent size: {self.latent_space_size}, Layers count: {self.number_of_layers}"

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
