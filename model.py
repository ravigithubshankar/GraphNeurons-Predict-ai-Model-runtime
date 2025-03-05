from tensorflow.keras import layers
class CNConv(layers.Layer):
    def __init__(self, num_filters, k, activation='relu', **kwargs):
        super(CNConv, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.k = k
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      (self.k, input_shape[-1], self.num_filters))

    def cheb_polynomials(self, laplacian, k):
        cheb_polys = [tf.eye(tf.shape(laplacian)[0])]  # T_0(x) = I
        cheb_polys.append(laplacian)  # T_1(x) = L
        for i in range(2, k):
            cheb_polys.append(2 * laplacian * cheb_polys[-1] - cheb_polys[-2])
        return cheb_polys

    def call(self, inputs, laplacian):
        laplacian = tf.convert_to_tensor(laplacian, tf.float32)
        graph_signal = inputs

        # Compute ChebNet polynomials
        cheb_polys = self.cheb_polynomials(laplacian, self.k)

        # ChebNet convolution
        cheb_filter = tf.linalg.matrix_transpose(self.kernel)
        cheb_conv = tf.matmul(tf.concat([tf.matmul(graph_signal, cheb_poly) for cheb_poly in cheb_polys], axis=-1), cheb_filter)
        output = self.activation(cheb_conv)

        return output

class SAGEConv(layers.Layer):
    def __init__(self, num_neighbors, num_filters, activation='relu', **kwargs):
        super(SAGEConv, self).__init__(**kwargs)
        self.num_neighbors = num_neighbors
        self.num_filters = num_filters
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      (input_shape[-1] * (self.num_neighbors + 1),
                                       self.num_filters))

    def call(self, inputs, adjacency_matrix):
        adjacency_matrix = tf.cast(adjacency_matrix, tf.float32)
        graph_signal = inputs

        # Aggregate neighbors using mean
        aggregated_neighbors = tf.matmul(adjacency_matrix, graph_signal) / (
                tf.reduce_sum(adjacency_matrix, axis=-1, keepdims=True) + 1e-7)

        # Concatenate aggregated neighbors and input features
        concatenated_features = tf.concat([graph_signal, aggregated_neighbors], axis=-1)

        # GraphSAGE convolution
        output = tf.matmul(concatenated_features, self.kernel)
        output = self.activation(output)

        return output
class GraphNeuralNetwork(tf.keras.Model):
    def __init__(self, ops, embed_dim):
        super(GraphNeuralNetwork, self).__init__()

        # Embedding layer for "op" node set
        self.embedding_layer = _OpEmbedding(ops, embed_dim)

        # ChebNet layer
        self.chebnet_layer = CNConv(num_filters=64, k=2)

        # SAGEConv layer
        self.sageconv_layer = SAGEConv(num_neighbors=3, num_filters=32)

        # Output layer (you may modify this based on your specific task)
        self.output_layer = tf.keras.layers.Dense(units=1, activation='linear')

    def call(self, graph, training=False):
        # Embedding layer
        graph_embedded_ops = self.embedding_layer(graph)

        # ChebNet layer
        config_adj = implicit.AdjacencyMultiplier(graph_embedded_ops, 'config')
        chebnet_output = self.chebnet_layer(graph_embedded_ops.node_sets['nconfig']['feats'], config_adj)

        # SAGEConv layer
        sageconv_output = self.sageconv_layer(chebnet_output, config_adj)

        # Global pooling (you may want to customize this based on your task)
        global_pooled = tf.reduce_mean(sageconv_output, axis=1)

        # Output layer
        model_output = self.output_layer(global_pooled)

        return model_output
