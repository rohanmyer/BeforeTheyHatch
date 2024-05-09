import tensorflow as tf
import numpy as np


def generate_heatmap(model, image):
    # build model from inputs to output of last convolution layer
    last_conv_layer = model.layers[0].get_layer("conv5_block3_out")
    last_conv_layer_model = tf.keras.Model(
        model.layers[0].inputs, last_conv_layer.output
    )

    # build model from output of last conv layer to final predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_ind in [-3, -2, -1]:
        x = model.layers[0].layers[layer_ind](x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # get gradients of prediction wrt conv output
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(image)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        preds_reshaped = tf.reshape(preds, [-1])  # Flatten all predictions
        # print("Reshaped predictions:", preds_reshaped.shape)
        top_pred_index = tf.argmax(preds_reshaped)
        top_class_channel = preds_reshaped[top_pred_index]
        # print("Top prediction index:", top_pred_index)

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply gradients by its feature map
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Average over all the filters to get a single 2D array
    gradcam = np.mean(last_conv_layer_output, axis=-1)
    # Clip the values (equivalent to applying ReLU) and normalize values
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)

    return gradcam


if __name__ == "__main__":
    from model import EmbryoClassifier

    model = EmbryoClassifier(num_classes=16)
    # model.load_weights("model")

    from tensorflow.keras.applications.resnet50 import preprocess_input

    image = np.random.rand(256, 256, 3)
    from PIL import Image

    picture = Image.fromarray((image * 255).astype(np.uint8))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    gradcam = generate_heatmap(model, image)

    scale = 256 / gradcam.shape[0]

    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    plt.imshow(picture.resize((256, 256)))
    plt.imshow(zoom(gradcam, zoom=(scale, scale)), alpha=0.5)
