import tensorflow as tf


class ImageModel(object):
    def __init__(self, label_len, image_height, image_width):
        self.model = None
        # Inputs and Labels
        self.num_label = label_len

        # Graph Initialize
        self.train_graph = tf.Graph()

        self.image_input = tf.placeholder(
            shape=(None, None, None, 3), dtype=tf.float32, name="image_input")
        print("Image Input Placeholder : " + str(self.image_input.shape))

        self.labels = tf.placeholder(
            shape=(None, 8), dtype=tf.int32, name="labels")
        print("Image labels Placeholder : " + str(self.labels.shape))

        self.resize_image = tf.image.resize_image_with_crop_or_pad(
            image=self.image_input, target_height=image_height, target_width=image_width)
        print("Image Resize : " + str(self.resize_image.shape))

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

            return lrelu

    def maxpool(self, input, scope):
        with tf.variable_scope(scope):
            return tf.nn.max_pool(name="maxpool", value=input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

    def flatten(self, tensor):
        output_shape = tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
        flatten = tf.reshape(
            tensor=tensor, shape=[-1, output_shape], name="flatten")
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

            return dropout

    # VGG16 Architecture
    def build_model(self):
        # Layer 1
        conv_1_1 = self.conv2d(self.resize_image, "conv_1_1",
                               self.resize_image.shape[-1], 64)
        conv_1_2 = self.conv2d(conv_1_1, "conv_1_2", 64, 64)
        maxpool_1 = self.maxpool(conv_1_2, "maxpool_1")
        dropout_1 = tf.nn.dropout(maxpool_1, keep_prob=0.5)

        # Layer 2
        conv_2_1 = self.conv2d(dropout_1, "conv_2_1", 64, 128)
        conv_2_2 = self.conv2d(conv_2_1, "conv_2_2", 128, 128)
        maxpool_2 = self.maxpool(conv_2_2, "maxpool_2")
        dropout_2 = tf.nn.dropout(maxpool_2, keep_prob=0.5)

        # Layer 3
        conv_3_1 = self.conv2d(dropout_2, "conv_3_1", 128, 256)
        conv_3_2 = self.conv2d(conv_3_1, "conv_3_2", 256, 256)
        conv_3_3 = self.conv2d(conv_3_2, "conv_3_3", 256, 256)
        maxpool_3 = self.maxpool(conv_3_3, "maxpool_3")
        dropout_3 = tf.nn.dropout(maxpool_3, keep_prob=0.5)

        # Layer 4
        conv_4_1 = self.conv2d(dropout_3, "conv_4_1", 256, 512)
        conv_4_2 = self.conv2d(conv_4_1, "conv_4_2", 512, 512)
        conv_4_3 = self.conv2d(conv_4_2, "conv_4_3", 512, 512)
        maxpool_4 = self.maxpool(conv_4_3, "maxpool_4")
        dropout_4 = tf.nn.dropout(maxpool_4, keep_prob=0.5)

        # Layer 5
        conv_5_1 = self.conv2d(dropout_4, "conv_5_1", 512, 512)
        conv_5_2 = self.conv2d(conv_5_1, "conv_5_2", 512, 512)
        conv_5_3 = self.conv2d(conv_5_2, "conv_5_3", 512, 512)
        maxpool_5 = self.maxpool(conv_5_3, "maxpool_5")
        dropout_5 = tf.nn.dropout(maxpool_5, keep_prob=0.5, name="dropout_5")

        dense_1 = self.dense(dropout_5, "dense_1", 4096)
        dense_2 = self.dense(dense_1, "dense_2", 4096)
        logits = tf.nn.dropout(dense_2, keep_prob=0.5, name="logits")

        predictions = tf.nn.softmax(logits=logits)

        return logits, predictions

    def loss(self, labels, logits):
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            return tf.reduce_mean(cross_entropy)

    def optimizer(self, loss, learning_rate=0.001):
        with tf.name_scope("optimizer"):
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    def accuracy(self):
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.labels, axis=-1))
            return tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def train_model(self, batches):
        loss_ = self.loss()
        optimizer_ = self.optimizer(loss_)
        accuracy_ = self.accuracy()

        init = tf.initialize_all_variables()
        with tf.Session(graph=self.train_graph) as sess:
            sess.run(init)

            for batch in batches:
                images = batch[0]
                labels = batch[1]
                feed_dict = {self.image_input: images, self.labels: labels}
                loss, opt, acc = sess.run(
                    [loss_, optimizer_, accuracy_], feed_dict=feed_dict)


def main():
    model = ImageModel(8, 1000, 754)
    model.build_model()


if __name__ == "__main__":
    main()
