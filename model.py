from argparse import ArgumentParser
import tensorflow as tf
import tensorflow.keras as keras
from preprocessing import data_load


class EmbryoClassifier(tf.keras.Model):

    def __init__(self, num_classes, input_shape=(256, 256, 3)):
        super(EmbryoClassifier, self).__init__()

        pretrained_model = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=input_shape,
            pooling="avg",
            classes=1000,
            weights="imagenet",
        )
        for layer in pretrained_model.layers:
            layer.trainable = False

        dropout1 = tf.keras.layers.Dropout(0.3)(pretrained_model.layers[-1].output)
        dense1 = tf.keras.layers.Dense(500, activation="relu")(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.4)(dense1)
        dense2 = tf.keras.layers.Dense(num_classes, activation="softmax")(dropout2)
        self.model = tf.keras.Model(inputs=pretrained_model.inputs, outputs=dense2)

    def call(self, inputs):
        return self.model(inputs)

    def save(self, path="model.weights.h5"):
        self.model.save_weights(path)

    def load(self, path="model.weights.h5"):
        self.model.load_weights(path)


def parseArguments():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--load_weights", action="store_true")  # load weights
    parser.add_argument("--batch_size", type=int, default=256)  # batch size
    parser.add_argument("--num_epochs", type=int, default=5)  # epochs
    parser.add_argument("--input_dim", type=int, default=256)  # input image dimension
    parser.add_argument("--learning_rate", type=float, default=1e-3)  # learning rate
    parser.add_argument("--num_classes", type=int, default=16)  # number of classes
    parser.add_argument(
        "--data_dir", type=str, default="./data"
    )  # directory of dataset folders
    args = parser.parse_args()
    return args


def main(args):

    train, val, test = data_load(args.data_dir, args.batch_size, args.input_dim)
    model = EmbryoClassifier(num_classes=args.num_classes)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # load weights
    path = ""
    if args.load_weights:
        raise NotImplementedError("Loading weights is not yet supported")
        from_pretrained_keras(path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # train model
    model.fit(
        train,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(val),
    )

    print("Evaluating model")
    model.evaluate(test)

    # save model
    # if not args.load_weights:
    #     push_to_hub_keras(model, path)
    print("Saving model")
    model.save()


if __name__ == "__main__":
    args = parseArguments()
    main(args)
