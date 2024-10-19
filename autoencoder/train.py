import os
import csv
import argparse
from glob import glob
from tqdm import tqdm
#
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboard.plugins import projector
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras.backend import get_graph
#
from network import AutoEncoder
from data_generator import DataGenerator, preprocess_input
from utils import read_config_file, indent_str, sample_reference_examples
#

parser = argparse.ArgumentParser(description='Run the training loop.')
parser.add_argument('-d', '--data_dir', type=str, help='Path to dataset.')
parser.add_argument('-c', '--config_path', type=str,
                    help='Path to config file.')
parser.add_argument('-o', '--output_dir', type=str,
                    help='Path to output directory.', default="out")
parser.add_argument('--lr', type=float,
                    help='The learning rate',
                    default=0.001)
parser.add_argument('--epochs', type=int,
                    help='The number of epochs to run training',
                    default=None)
parser.add_argument('--batch_size', type=int,
                    help='The samples per batch to train',
                    default=1)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

config = read_config_file(args.config_path)
samples_to_visualize = sorted(glob(os.path.join(args.data_dir, "*.jpg")))
training_loss_csv = os.path.join(args.output_dir, "training_loss.csv")
#
model = AutoEncoder(
    input_size=(config["input_size"], config["input_size"]),
    embedding_dims=config["embedding_dims"],
    encoder_hidden_dims=config["encoder_hidden_dims"],
    decoder_hidden_dims=config["decoder_hidden_dims"],
)
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
#
training_set = DataGenerator(
    data_dir=args.data_dir,
    shuffle=True,
    batch_size=args.batch_size,
    input_size=config["input_size"],
)


def setup_tensorboard():
    train_log_dir = os.path.join(args.output_dir, "logs", "train")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # setup embeddings view
    projector_config = projector.ProjectorConfig()
    embedding = projector_config.embeddings.add()
    embedding.tensor_name = "embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(train_log_dir, projector_config)
    with open(os.path.join(train_log_dir, "metadata.tsv"), "w") as outfile:
        for sample in samples_to_visualize:
            label = os.path.basename(sample).split("_")[0]
            outfile.write(f"{label}\n")

    return train_summary_writer


def train_step(samples):
    with tf.GradientTape() as tape:
        y_pred = model(samples, training=True)
        loss = tf.keras.losses.MSE(samples, y_pred)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)

    train_loss(loss)


def update_reference_examples():
    reference_examples = sample_reference_examples(data_dir=args.data_dir)
    reference_examples = np.array([
        preprocess_input(cv2.imread(img), input_size=config["input_size"])
        for img in reference_examples
    ])
    reference_examples_reconstructed = model(reference_examples)

    return [
        np.expand_dims(np.concatenate([t, r], axis=-1), axis=-1)
        for t, r in zip(reference_examples, reference_examples_reconstructed)
    ]


def update_learned_embeddings():
    encoder = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('encoder').output,
    )

    embeddings = []
    for sample in samples_to_visualize:
        x = cv2.imread(sample)
        x = preprocess_input(
            x,
            input_size=config["input_size"]
        )
        x = np.expand_dims(x, axis=0)
        embedding = encoder(x)
        embeddings.append(embedding[0])

    embeddings = np.array(embeddings)

    embeddings_ckpt.embeddings.assign(tf.Variable(embeddings))


def plot_loss_graph():
    df = pd.read_csv(training_loss_csv)
    _ = plt.figure(figsize=(5, 5))
    plt.plot(df['epoch'], df['loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(args.output_dir, "loss_plot.png"))


if __name__ == "__main__":

    train_summary_writer = setup_tensorboard()

    current_epoch = tf.Variable(0)
    best_epoch = tf.Variable(0)
    best_loss = tf.Variable(99999, dtype=tf.float32)
    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=current_epoch,
        best_loss=best_loss,
        best_epoch=best_epoch,
    )
    manager = tf.train.CheckpointManager(
        ckpt,
        os.path.join(args.output_dir),
        max_to_keep=3
    )
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored checkpoint from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    embeddings = tf.Variable(
        np.zeros((len(samples_to_visualize), config["embedding_dims"])),
        dtype=tf.float32
    )
    embeddings_ckpt = tf.train.Checkpoint(embeddings=embeddings)
    embeddings_manager = tf.train.CheckpointManager(
        embeddings_ckpt,
        os.path.join(args.output_dir, "logs", "train"),
        max_to_keep=1,
        checkpoint_name="embeddings",
    )
    embeddings_ckpt.restore(embeddings_manager.latest_checkpoint)

    while True:
        curr_epoch = int(ckpt.epoch)
        if args.epochs is not None and curr_epoch > args.epochs:
            break

        for i in tqdm(range(len(training_set))):
            samples = training_set[i]
            train_step(samples)

        if float(train_loss.result()) <= float(ckpt.best_loss):
            ckpt.best_loss.assign(train_loss.result())
            ckpt.best_epoch.assign(curr_epoch)
            print("Saving new best")
            model.save(os.path.join(args.output_dir, "model.keras"))

        print(f"Epoch: {curr_epoch}, Loss: {float(train_loss.result())}")

        # write to csv
        pd.DataFrame({
            "epoch": [curr_epoch],
            "loss": [float(train_loss.result())],
        }).to_csv(training_loss_csv, mode='a', index=False, header=curr_epoch == 0)

        reference_examples = update_reference_examples()

        if curr_epoch % 10 == 0:
            cv2.imwrite(
                os.path.join(args.output_dir, "reference_examples.jpg"),
                (np.concatenate(reference_examples, axis=0)*255).astype(np.uint8)
            )
            plot_loss_graph()
            update_learned_embeddings()
            embeddings_manager.save()

        ckpt.epoch.assign_add(1)

        with train_summary_writer.as_default():
            tf.summary.image(
                'reference_images',
                reference_examples,
                max_outputs=len(reference_examples),
                step=curr_epoch
            )
            tf.summary.scalar(
                'loss',
                train_loss.result(),
                step=curr_epoch
            )

        manager.save()
