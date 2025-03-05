
#The _mlp function is a utility to create a Multi-Layer Perceptron (MLP)
def _mlp(dims, hidden_activation, l2reg=1e-4, use_bias=True):
  """Helper function for multi-layer perceptron (MLP)."""
  layers = []
  for i, dim in enumerate(dims):
    if i > 0:
      layers.append(tf.keras.layers.Activation(hidden_activation))
    layers.append(tf.keras.layers.Dense(
        dim, kernel_regularizer=tf.keras.regularizers.l2(l2reg),
        use_bias=use_bias))
  return tf.keras.Sequential(layers)

#The _OpEmbedding class is a model designed to embed specific nodes (from the GraphTensor) into a feature space. 
class _OpEmbedding(tf.keras.Model):
  """Embeds GraphTensor.node_sets['op']['op'] nodes into feature 'op_e'."""

  def __init__(self, num_ops: int, embed_d: int, l2reg: float = 1e-4):
    super().__init__()
    self.embedding_layer = tf.keras.layers.Embedding(
        num_ops, embed_d, activity_regularizer=tf.keras.regularizers.l2(l2reg))

  def call(
      self, graph: tfgnn.GraphTensor,
      training: bool = False) -> tfgnn.GraphTensor:
    op_features = dict(graph.node_sets['op'].features)
    op_features['op_e'] = self.embedding_layer(
        tf.cast(graph.node_sets['op']['op'], tf.int32))
    return graph.replace_features(node_sets={'op': op_features})

