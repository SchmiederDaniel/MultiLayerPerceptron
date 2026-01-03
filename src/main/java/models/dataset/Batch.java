package models.dataset;

import neuralnetwork.math.Tensor;

public class Batch {
    public final Tensor[] inputs;
    public final Tensor[] targets;
    
    Batch(Tensor[] inputs, Tensor[] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }
}