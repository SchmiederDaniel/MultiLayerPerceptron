package neuralnetwork.layer;

import neuralnetwork.math.Tensor;

/**
 * A layer that flattens any input tensor (rank > 1) into a 1D Vector.
 * This is typically used to connect the output of a Convolutional or Pooling layer
 * to a subsequent Dense layer.
 * * It performs no operation other than reshaping and is used purely for compatibility.
 */
public class Flatten extends Layer {
    
    // The input tensor's shape is cached to know how to reshape the gradient in backward pass.
    private int[] lastInputShape;

    public Flatten() {
        // No parameters or complex initialization needed
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.rank() < 2) {
            // No need to flatten a Scalar (rank 0) or a Vector (rank 1), just return a copy
            return input.deepCopy();
        }
        
        // Cache the input shape for the backward pass
        this.lastInputShape = input.shape();
        
        // Use the Tensor base class's flatten() method to convert it to a 1D Vector
        return input.flatten();
    }

    @Override
    public Tensor backward(Tensor outputGradient, float learningRate) {
        // The gradient flowing back (outputGradient) is a Vector (dL/dVector).
        // It must be reshaped back into the original input shape (dL/d3DTensor).
        
        if (lastInputShape == null) {
            throw new IllegalStateException("Flatten layer backward pass called before forward pass.");
        }
        
        // The total size remains the same, so we simply reshape the gradient 
        // using the cached shape from the forward pass.
        // The reshape operation uses the same row-major order as the flatten operation.
        return outputGradient.reshape(lastInputShape);
    }

    @Override
    public Flatten clone() {
        // The layer has no state (weights/bias) other than cached shape, 
        // so a simple new instance is sufficient.
        Flatten cloned = new Flatten();
        // lastInputShape is not carried over, as is standard for layer caching.
        cloned.isTraining = this.isTraining;
        return cloned;
    }
}