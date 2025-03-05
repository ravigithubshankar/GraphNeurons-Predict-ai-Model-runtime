LAYOUT_DATA_ROOT = '/kaggle/input/predict-ai-model-runtime/npz_all/npz/layout'
SOURCE = 'xla'  # Can be "xla" or "nlp"
SEARCH = 'random'  # Can be "random" or "default"

# Batch size information.
BATCH_SIZE = 16  # Number of graphs per batch.
CONFIGS_PER_GRAPH = 5  # Number of configurations (features and target values) per graph.
MAX_KEEP_NODES = 1000

import os
layout_data_root_dir = os.path.join(
      os.path.expanduser(LAYOUT_DATA_ROOT), SOURCE, SEARCH)

layout_npz_dataset = layout_data.get_npz_dataset(
    layout_data_root_dir,
    min_train_configs=CONFIGS_PER_GRAPH,
    max_train_configs=500,  # If any graph has more than this configurations, it will be filtered [speeds up loading + training]
    cache_dir='cache'
)

def pair_layout_graph_with_label(graph: tfgnn.GraphTensor):
    
    label=tf.cast(graph.node_sets["g"]["runtimes"],tf.float32)/1e7
    return graph,label

