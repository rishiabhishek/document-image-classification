import tensorflow as tf
from datetime import datetime
import os


class ImageModel(object):
    def __init__(self, num_labels, image_height, image_width):
        # Inputs and Labels
        self.num_labels = num_labels

        self.image_input = tf.placeholder(
            shape=(None, image_height, image_width, 3), dtype=tf.float32, name="image_input")
        print("Image Input Placeholder : " + str(self.image_input.shape))

        self.labels = tf.placeholder(
            shape=(None, self.num_labels), dtype=tf.int32, name="labels")
        print("Image labels Placeholder : " + str(self.labels.shape))

        # self.resize_image = tf.image.resize_image_with_crop_or_pad(
        #     image=self.image_input, target_height=image_height, target_width=image_width)
        # print("Image Resize : " + str(self.resize_image.shape))

    def conv2d(self, input, scope, in_channels, out_channels, alpha=0.3):
        with tf.variable_scope(scope):
            conv_weights = tf.get_variable(name="conv_weights", shape=(
                3, 3, in_channels, out_channels), initializer=tf.truncated_normal_initializer())

            conv = tf.nn.conv2d(name="conv", input=input, filter=conv_weights, strides=(
                1, 2, 2, 1), padding="SAME")

            bias_weights = tf.get_variable(
                name="bias_weights", shape=out_channels, initializer=tf.zeros_initializer())

            bias = tf.nn.bias_add(conv, bias_weights)

            lrelu = tf.nn.leaky_relu(bias, alpha=alpha, name="lrelu")

            tf.summary.histogram('lrelu', lrelu)
            return lrelu

    def maxpool(self, input, scope):
        with tf.variable_scope(scope):
            maxpool = tf.nn.max_pool(name="maxpool", value=input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                                     padding="SAME")
            tf.summary.histogram('maxpool', maxpool)
            return maxpool

    def flatten(self, tensor):
        output_shape = tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
        flatten = tf.reshape(
            tensor=tensor, shape=[-1, output_shape], name="flatten")
        tf.summary.histogram('flatten', flatten)
        return flatten

    def dense(self, tensor, name, output_size):
        with tf.variable_scope(name):
            weights = tf.get_variable(name="dense_weight", shape=(
                tensor.shape[-1], output_size), initializer=tf.truncated_normal_initializer())

            bias = tf.get_variable(
                name="dense_bias", shape=output_size, initializer=tf.zeros_initializer())

            dense = tf.add(tf.matmul(tensor, weights,
                                     name="matmul"), bias, name="add_bias")

            dropout = tf.nn.dropout(dense, keep_prob=0.5)
            tf.summary.histogram('dropout', dropout)
            return dropout

    def build_model(self):

        # Layer 0
        conv_0_1 = self.conv2d(self.image_input, "conv_0_1",
                               self.image_input.shape[-1], 20)
        conv_0_2 = self.conv2d(conv_0_1, "conv_0_2", 20, 20)
        maxpool_0 = self.maxpool(conv_0_2, "maxpool_0")
        dropout_0 = tf.nn.dropout(maxpool_0, keep_prob=0.5, name="dropout_0")
        print("Layer 0 : " + str(dropout_0.shape))

        # Layer 1
        conv_1_1 = self.conv2d(dropout_0, "conv_1_1", 20, 64)
        conv_1_2 = self.conv2d(conv_1_1, "conv_1_2", 64, 64)
        maxpool_1 = self.maxpool(conv_1_2, "maxpool_1")
        dropout_1 = tf.nn.dropout(maxpool_1, keep_prob=0.5, name="dropout_1")
        print("Layer 1 : " + str(dropout_1.shape))

        # Layer 2
        conv_2_1 = self.conv2d(dropout_1, "conv_2_1", 64, 128)
        conv_2_2 = self.conv2d(conv_2_1, "conv_2_2", 128, 128)
        maxpool_2 = self.maxpool(conv_2_2, "maxpool_2")
        dropout_2 = tf.nn.dropout(maxpool_2, keep_prob=0.5, name="dropout_2")
        print("Layer 2 : " + str(dropout_2.shape))

        # Layer 3
        conv_3_1 = self.conv2d(dropout_2, "conv_3_1", 128, 256)
        conv_3_2 = self.conv2d(conv_3_1, "conv_3_2", 256, 256)
        conv_3_3 = self.conv2d(conv_3_2, "conv_3_3", 256, 256)
        maxpool_3 = self.maxpool(conv_3_3, "maxpool_3")
        dropout_3 = tf.nn.dropout(maxpool_3, keep_prob=0.5, name="dropout_3")
        print("Layer 3 : " + str(dropout_3.shape))

        # Layer 4
        conv_4_1 = self.conv2d(dropout_3, "conv_4_1", 256, 512)
        conv_4_2 = self.conv2d(conv_4_1, "conv_4_2", 512, 512)
        conv_4_3 = self.conv2d(conv_4_2, "conv_4_3", 512, 512)
        maxpool_4 = self.maxpool(conv_4_3, "maxpool_4")
        dropout_4 = tf.nn.dropout(maxpool_4, keep_prob=0.5, name="dropout_4")
        print("Layer 4 : " + str(dropout_4.shape))

        # Layer 5
        conv_5_1 = self.conv2d(dropout_4, "conv_5_1", 512, 512)
        conv_5_2 = self.conv2d(conv_5_1, "conv_5_2", 512, 512)
        conv_5_3 = self.conv2d(conv_5_2, "conv_5_3", 512, 512)
        maxpool_5 = self.maxpool(conv_5_3, "maxpool_5")
        dropout_5 = tf.nn.dropout(maxpool_5, keep_prob=0.5, name="dropout_5")
        print("Layer 5 : " + str(dropout_5.shape))

        flatten = self.flatten(dropout_5)
        print("Flatten : " + str(flatten.shape))

        dense_1 = self.dense(flatten, "dense_1", 4096)
        dense_2 = self.dense(dense_1, "dense_2", self.num_labels)
        logits = tf.nn.dropout(dense_2, keep_prob=0.5, name="logits")
        print("Dense : " + str(logits.shape))

        predictions = tf.nn.softmax(logits=logits)
        print("Softmax : " + str(predictions.shape))

        tf.summary.histogram('predictions', predictions)
        return logits, predictions

    def loss(self, logits):
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
            tf.summary.scalar('loss', loss)
            return loss

    def optimizer(self, cost, learning_rate=0.00001):
        with tf.name_scope("optimizer"):
            # opt = tf.train.AdamOptimizer().minimize(loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gvs = optimizer.compute_gradients(cost)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs)

            return train_op

    def accuracy(self, predictions):
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(predictions, axis=-1), tf.argmax(self.labels, axis=-1))
            acc = tf.reduce_mean(tf.cast(correct_predictions, "float32"), name="accuracy")

            tf.summary.scalar('accuracy', acc)
            return acc

    def train_model(self, imageDataset, batch_size=20, epochs=20):
        logits, predictions = self.build_model()
        loss_ = self.loss(logits)
        optimizer_ = self.optimizer(loss_)
        accuracy_ = self.accuracy(predictions)

        logdir = "/Volumes/My Passport/abhishek/Datasets/Image Dataset/rvl-cdip/log_dir" + '/' + datetime.now().strftime(
            '%Y%m%d-%H%M%S') + '/'

        # Operation merging summary data for TensorBoard
        summary = tf.summary.merge_all()

        # Define saver to save model state at checkpoints
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            summary_writer = tf.summary.FileWriter(logdir, sess.graph)
            for i in range(epochs):
                print("Epochs : " + str(i))

                batch_count = 0
                hasNext = True
                while hasNext:
                    batches = imageDataset.build_dataset(batch_size, "train")
                    if batches:
                        batch = next(batches)
                        images = batch[0]
                        labels = batch[1]
                        feed_dict = {self.image_input: images, self.labels: labels}
                        batch_count += 1
                        _, loss, acc = sess.run([optimizer_, loss_, accuracy_], feed_dict=feed_dict)
                        print(
                            "Batch : {0} \t  ----  Loss : {1:.2f} \t ---- Accuracy : {2:.2f}".format(batch_count, loss,
                                                                                                     acc))

                        # Saving Summary
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i)

                        if batch_count % 50 == 0:
                            checkpoint_file = os.path.join(logdir, 'checkpoint')
                            saver.save(sess, checkpoint_file, global_step=i)
                            print('Saved checkpoint')
                    else:
                        hasNext = False


def main():
    model = ImageModel(8, 1000, 754)
    model.build_model()


if __name__ == "__main__":
    main()
