import tqdm
_INFERENCE_CONFIGS_BATCH_SIZE = 50

output_csv_filename = f'inference_layout_{SOURCE}_{SEARCH}.csv'
print('\n\n   Running inference on test set ...\n\n')
test_rankings = []

assert layout_npz_dataset.test.graph_id is not None
for graph in tqdm.tqdm(layout_npz_dataset.test.iter_graph_tensors(),
                       total=layout_npz_dataset.test.graph_id.shape[-1],
                       desc='Inference'):
    num_configs = graph.node_sets['g']['runtimes'].shape[-1]
    all_scores = []
    for i in tqdm.tqdm(range(0, num_configs, _INFERENCE_CONFIGS_BATCH_SIZE)):
        end_i = min(i + _INFERENCE_CONFIGS_BATCH_SIZE, num_configs)
        # Take a cut of the configs.
        node_set_g = graph.node_sets['g']
        subconfigs_graph = tfgnn.GraphTensor.from_pieces(
            edge_sets=graph.edge_sets,
            node_sets={
                'op': graph.node_sets['op'],
                'nconfig': tfgnn.NodeSet.from_fields(
                    sizes=graph.node_sets['nconfig'].sizes,
                    features={
                        'feats': graph.node_sets['nconfig']['feats'][:, i:end_i],
                    }),
                'g': tfgnn.NodeSet.from_fields(
                    sizes=tf.constant([1]),
                    features={
                        'graph_id': node_set_g['graph_id'],
                        'runtimes': node_set_g['runtimes'][:, i:end_i],
                        'kept_node_ratio': node_set_g['kept_node_ratio'],
                    })
            })
        h = model.forward(subconfigs_graph, num_configs=(end_i - i),
                          backprop=False)
        all_scores.append(h[0])
    all_scores = tf.concat(all_scores, axis=0)
    graph_id = graph.node_sets['g']['graph_id'][0].numpy().decode()
    sorted_indices = tf.strings.join(
        tf.strings.as_string(tf.argsort(all_scores)), ';').numpy().decode()
    test_rankings.append((graph_id, sorted_indices))

with tf.io.gfile.GFile(output_csv_filename, 'w') as fout:
    fout.write('ID,TopConfigs\n')
    for graph_id, ranks in test_rankings:
        fout.write(f'layout:{SOURCE}:{SEARCH}:{graph_id},{ranks}\n')
print('\n\n   ***  Wrote', output_csv_filename, '\n\n')
