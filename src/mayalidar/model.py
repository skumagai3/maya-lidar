"""Keras 3 models, loss and metric for three-class multilabel segmentation.

Classes can overlap (buildings often sit on platforms), so the output layer has one
independent sigmoid per class rather than a softmax. The sigmoid outputs are scores
in (0, 1); they are not calibrated probabilities.
"""
from __future__ import annotations

import keras
from keras import layers, ops

from .data import CLASSES


def conv_block(x, filters, dropout=0.0):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    if dropout:
        x = layers.SpatialDropout2D(dropout)(x)
    return x


def up_block(x, skip, filters):
    x = layers.UpSampling2D(2, interpolation="bilinear")(x)
    if skip is not None:
        x = layers.Concatenate()([x, skip])
    return conv_block(x, filters)


def unet(input_shape=(480, 480, 3), n_classes=len(CLASSES), base_filters=32, depth=4, dropout=0.0):
    """Plain 2D U-Net. 480 is divisible by 2**4, so depth <= 4 keeps shapes exact.
    dropout > 0 adds SpatialDropout2D after the bottleneck block."""
    inp = keras.Input(input_shape)
    x, skips = inp, []
    for d in range(depth):
        x = conv_block(x, base_filters * 2 ** d)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    x = conv_block(x, base_filters * 2 ** depth, dropout=dropout)
    for d in reversed(range(depth)):
        x = up_block(x, skips[d], base_filters * 2 ** d)
    out = layers.Conv2D(n_classes, 1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inp, out, name="unet")


RESNET50_SKIPS = ("conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out")


def resnet50_unet(input_shape=(480, 480, 3), n_classes=len(CLASSES), weights="imagenet",
                  freeze_encoder=True):
    """U-Net decoder on an ImageNet-pretrained ResNet50 encoder. The encoder
    downsamples by 32 and 480 / 32 = 15, so shapes stay exact. Inputs are the three
    standardized relief channels, not RGB. Train with the encoder frozen first,
    then call `unfreeze` and fine-tune at a lower learning rate."""
    enc = keras.applications.ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    enc.trainable = not freeze_encoder
    skips = [enc.get_layer(n).output for n in RESNET50_SKIPS]  # strides 2, 4, 8, 16
    x = enc.output                                              # stride 32
    for skip, f in zip(reversed(skips), (256, 128, 64, 32)):
        x = up_block(x, skip, f)
    x = up_block(x, None, 16)
    out = layers.Conv2D(n_classes, 1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(enc.input, out, name="resnet50_unet")


def unfreeze(model, from_layer="conv4_block1_1_conv"):
    """Make encoder layers trainable from from_layer onward. Batch-norm layers stay
    frozen, which is the usual practice when fine-tuning with small batches."""
    on = False
    for layer in model.layers:
        if layer.name == from_layer:
            on = True
        if on and not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True


def dice_bce_loss(class_weights=(1.0, 1.0, 1.0), bce_weight=1.0, smooth=1.0, skip_absent_classes=False):
    """bce_weight * BCE + weighted mean over classes of (1 - soft Dice).

    BCE is averaged over pixels and classes. Soft Dice is computed per class over
    the whole batch. With the defaults this is the common unweighted BCE + Dice.

    In a batch with no labeled pixels of class c, its Dice term reduces to
    1 - smooth / (sum of its sigmoid outputs + smooth), which can only be lowered by
    pushing every output for c toward zero. With skip_absent_classes=True, such
    classes are left out of the Dice average for that batch (BCE still covers
    them). If no class is present, the Dice part is zero."""
    w = ops.convert_to_tensor(class_weights, "float32")

    def loss(y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        bce = ops.mean(keras.losses.binary_crossentropy(y_true, y_pred))
        inter = ops.sum(y_true * y_pred, axis=(0, 1, 2))
        denom = ops.sum(y_true + y_pred, axis=(0, 1, 2))
        dice = 1.0 - (2.0 * inter + smooth) / (denom + smooth)
        wc = w
        if skip_absent_classes:
            present = ops.cast(ops.sum(y_true, axis=(0, 1, 2)) > 0, "float32")
            wc = w * present
        dice_mean = ops.sum(wc * dice) / ops.maximum(ops.sum(wc), 1e-7)
        return bce_weight * bce + dice_mean

    loss.__name__ = "dice_bce"
    return loss


class ClassIoU(keras.metrics.Metric):
    """Intersection over union for one class, with TP, FP and FN accumulated over
    every pixel seen in the epoch (not averaged per tile). This is how the
    Chactun challenge was scored. Predictions are sigmoid output > threshold."""

    def __init__(self, class_idx, threshold=0.5, name=None, **kwargs):
        super().__init__(name=name or f"iou_{CLASSES[class_idx]}", **kwargs)
        self.class_idx, self.threshold = class_idx, threshold
        self.inter = self.add_weight(name="inter", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = ops.cast(y_true[..., self.class_idx] > 0.5, "float32")
        p = ops.cast(y_pred[..., self.class_idx] > self.threshold, "float32")
        self.inter.assign_add(ops.sum(t * p))
        self.union.assign_add(ops.sum(ops.maximum(t, p)))

    def result(self):
        return self.inter / (self.union + 1e-7)

    def reset_state(self):
        self.inter.assign(0.0)
        self.union.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "class_idx": self.class_idx, "threshold": self.threshold}


class MeanIoU(keras.metrics.Metric):
    """Unweighted mean over classes of the per-class IoU, each accumulated over
    the epoch as in ClassIoU. This is the challenge's mean IoU, and the score
    used to select the best checkpoint."""

    def __init__(self, threshold=0.5, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.inter = self.add_weight(name="inter", shape=(len(CLASSES),), initializer="zeros")
        self.union = self.add_weight(name="union", shape=(len(CLASSES),), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = ops.cast(y_true > 0.5, "float32")
        p = ops.cast(y_pred > self.threshold, "float32")
        self.inter.assign_add(ops.sum(t * p, axis=(0, 1, 2)))
        self.union.assign_add(ops.sum(ops.maximum(t, p), axis=(0, 1, 2)))

    def result(self):
        return ops.mean(self.inter / (self.union + 1e-7))

    def reset_state(self):
        self.inter.assign(ops.zeros_like(self.inter))
        self.union.assign(ops.zeros_like(self.union))

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}


def iou_metrics(threshold=0.5):
    return [ClassIoU(i, threshold) for i in range(len(CLASSES))] + [MeanIoU(threshold)]
