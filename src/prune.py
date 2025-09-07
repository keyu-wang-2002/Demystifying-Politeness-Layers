def prune_selected_layers(model, prune_list):
    num_layers = len(model.model.layers)
    layers_to_remove = prune_list
    remove_n_layers = len(layers_to_remove)

    if remove_n_layers <= 0:
        return model
    if remove_n_layers >= num_layers:
        raise ValueError(f"Cannot remove all layers. Model has {num_layers} layers, attempted to remove {remove_n_layers}")

    layers_to_remove.sort(reverse=True)  

    print("remove layers: " + str(layers_to_remove))

    for index in layers_to_remove:
        del model.model.layers[index]

    model.config.num_hidden_layers = len(model.model.layers)
    
    return model