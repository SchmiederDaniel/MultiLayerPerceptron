package neuralnetwork.layer;

import neuralnetwork.math.Tensor;

/**
 * A layer that changes the shape of the input tensor without changing its data or total size.
 * This is useful for reshaping 1D vectors back into 3D volumes or rearranging dimensions.
 */
public class Reshape extends Layer {
    
    // The target shape for the output tensor
    private final int[] targetShape;
    
    // The input shape is cached for the backward pass
    private int[] lastInputShape;

    /**
     * Constructs a Reshape layer.
     * @param targetShape The desired output shape (e.g., {channels, height, width}).
     * The product of dimensions must equal the total size of the input tensor.
     */
    public Reshape(int... targetShape) {
        if (targetShape == null || targetShape.length == 0) {
            throw new IllegalArgumentException("Target shape must be specified.");
        }
        this.targetShape = targetShape;
    }

    @Override
    public Tensor forward(Tensor input) {
        // Cache the input shape for the backward pass
        this.lastInputShape = input.shape();
        
        // Use the Tensor base class's reshape() method to change the shape
        // The base method handles the check that input.size() == product(targetShape)
        return input.reshape(targetShape);
    }

    @Override
    public Tensor backward(Tensor outputGradient, float learningRate) {
        if (lastInputShape == null) {
            throw new IllegalStateException("Reshape layer backward pass called before forward pass.");
        }
        
        // The backward pass requires reshaping the gradient back to the original input shape.
        // The gradient flowing back (outputGradient) is already correctly ordered (C-order/row-major),
        // it just needs its dimensions restored.
        return outputGradient.reshape(lastInputShape);
    }

    @Override
    public Reshape clone() {
        // The layer has no trainable parameters.
        Reshape cloned = new Reshape(targetShape.clone());
        cloned.isTraining = this.isTraining;
        return cloned;
    }
}