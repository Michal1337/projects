import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        # based on https://github.com/openai/whisper/blob/main/whisper/model.py#L162
        self.conv1 = tf.keras.layers.Conv1D(
            d_model, kernel_size=3, padding="same", activation=tf.keras.activations.gelu
        )
        self.conv2 = tf.keras.layers.Conv1D(
            d_model,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=tf.keras.activations.gelu,
        )
        self.pos_encoding = tf.keras.layers.Embedding(block_size, d_model)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        length = tf.shape(x)[1]
        x = x + self.pos_encoding(tf.range(length))
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class SelfAttention(BaseAttention):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1
    ):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = SelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.causal_self_attention(x=x)
        x = self.ffn(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        block_size: int,
        dropout_rate: float = 0.1,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(block_size=block_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x)

        return x


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        block_size: int,
        dropout_rate: float = 0.1,
        num_classes: int = 30,
    ):
        super().__init__()

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            block_size=block_size,
            dropout_rate=dropout_rate,
        )

        self.final_layer = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.decoder(x)
        x = x[:, -1, :]
        logits = self.final_layer(x)
        return logits
