layout_train=(
    layout_npz_dataset.train.get_graph_tensors_dataset(
        CONFIGS_PER_GRAPH,max_nodes=MAX_KEEP_NODES)
        .shuffle(100,reshuffle_each_iteration=True)
        .batch(BATCH_SIZE,drop_remainder=False)
        .map(tfgnn.GraphTensor.merge_batch_to_components)
        .map(pair_layout_graph_with_label))
layout_valid=(
    layout_npz_dataset.validation.get_graph_tensors_dataset(
        CONFIGS_PER_GRAPH)
        .batch(BATCH_SIZE,drop_remainder=False)
        .map(tfgnn.GraphTensor.merge_batch_to_components)
        .map(pair_layout_graph_with_label))

graph_batch,config_runtimes=next(iter(layout_train.take(1)))
print("graph_batch=")
print(graph_batch)
print("\n\n")
print("config_runtimes=")
print(config_runtimes)

print('graph_batch.context =', graph_batch.context)
# Note: graph_batch.context.sizes must be equal to BATCH_SIZE.
# Lets print-out all features for all nodesets.

for node_set_name in sorted(graph_batch.node_sets.keys()):
    print(f'\n\n #####  NODE SET "{node_set_name}" #########')
    print('** Has sizes: ', graph_batch.node_sets[node_set_name].sizes)
    for feature_name in graph_batch.node_sets[node_set_name].features.keys():
        print(f'\n Feature "{feature_name}" has values')
        print(graph_batch.node_sets[node_set_name][feature_name])


print('\n config edge set: ', graph_batch.edge_sets['config'])  
print('\n config source nodes: ', graph_batch.edge_sets['config'].adjacency.source)
print('\n config target nodes: ', graph_batch.edge_sets['config'].adjacency.target)
print('\n g_op edge set: ', graph_batch.edge_sets['g_op'])
print('\n g_config edge set: ', graph_batch.edge_sets['g_config'])


print(graph_batch.edge_sets['config'])
print(graph_batch.edge_sets['config'].adjacency.source)
print(graph_batch.edge_sets['config'].adjacency.target)

ops=layout_npz_dataset.num_ops
print(f"number of ops in the dataset:",{ops})
embedding_layer = _OpEmbedding(ops,16)
graph_batch_embedded_ops=embedding_layer(graph_batch)
print(f"\n\n Before embedding,node-set op=\n' ,{graph_batch.node_sets['op']}")
print(f"\n\n After embedding,node-set op=\n' ,{graph_batch_embedded_ops.node_sets['op']}")

op_e=graph_batch_embedded_ops.node_sets['op']['op_e']
config_features=graph_batch_embedded_ops.node_sets['nconfig']['feats']
print(f"op_e.shape ==  {op_e.shape}")
print(f"config_feature.shape == {config_features.shape}")

config_adj = implicit.AdjacencyMultiplier(graph_batch_embedded_ops, 'config')
print('config_adj.shape =', config_adj.shape)
resized_config_features = config_adj @ config_features
print('resized_config_features.shape =', resized_config_features.shape)

