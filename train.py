from model import SAGEConv,GraphNeuralNetwork,Cheb_ploynomial
gnn_model = GraphNeuralNetwork(ops=ops, embed_dim=16)
gnn_model.compile(optimizer='adam', loss='mse')

loss=tfr.keras.losses.ListMLELoss()
opt=tf._optimizers.Adam(learning_rate=0.01,clipnorm=0.5)
model.compile(loss=loss,optimizer=opt,metrics=[
    tfr.keras.metrics.OPAMetric(name="opa_metric"),
])

early_stopping=10
best_params=None
best_val_opa=-1
best_val_at_epoch=-1
epochs=2

for i in range(epochs):
   # history = model.fit(
        #layout_train, epochs=1, verbose=1, validation_data=layout_valid,
        #validation_freq=1)
    features=torch.tensor(node_ftr,dtype=torch.float32).to(device)
    labels=torch.tensor(y,dtype=torch.long).to(device)
    model.to(device)
    model.train()
    with torch.set_grad_enabled(True):
    optimizer.zero_grad()
    output, edge_weights = model(features, edge_index, edgenet_input,random_walk_embeddings)
    loss_train = torch.nn.CrossEntropyLoss()(output[train_ind], labels[train_ind])
    loss_train.backward()
    optimizer.step()

    train_loss = history.history['loss'][-1]
    train_opa = history.history['opa_metric'][-1]
    val_loss = history.history['val_loss'][-1]
    val_opa = history.history['val_opa_metric'][-1]
    if val_opa > best_val_opa:
        best_val_opa = val_opa
        best_val_at_epoch = i
        best_params = {v.ref: v + 0 for v in model.trainable_variables}
        print(' * [@%i] Validation (NEW BEST): %s' % (i, str(val_opa)))
    elif early_stopping > 0 and i - best_val_at_epoch >= early_stopping:
      print('[@%i] Best accuracy was attained at epoch %i. Stopping.' % (i, best_val_at_epoch))
      break



# Restore best parameters.
print('Restoring parameters corresponding to the best validation OPA.')
assert best_params is not None
for v in model.trainable_variables:
    v.assign(best_params[v.ref])
