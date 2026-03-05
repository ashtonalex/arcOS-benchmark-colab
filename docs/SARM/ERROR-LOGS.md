---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_207/885911301.py in <cell line: 0>()
      4 print('Building HeteroGNN model...\n')
      5 
----> 6 hetero_model = HeteroGNNModel.build_from_checkpoint_or_train(
      7     config=config,
      8     retriever=retriever,

/content/arcOS-benchmark-colab/src/gnn/hetero_model_wrapper.py in build_from_checkpoint_or_train(cls, config, retriever, train_samples, val_samples, scene_graphs)
     86                 )
     87                 if len(train_pyg) == 0:
---> 88                     raise ValueError(
     89                         f"No valid training examples found from {len(train_samples)} samples. "
     90                         "This usually means object_names is missing from PCST subgraphs. "

ValueError: No valid training examples found from 3473 samples. This usually means object_names is missing from PCST subgraphs. Delete stale hetero_pyg_train.pkl / hetero_pyg_val.pkl from Drive and retry.