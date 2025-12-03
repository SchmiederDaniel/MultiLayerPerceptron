package neuralnetwork.layer;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.math.NumpyArray;

public class FourierTransform extends Layer {
  private int fourierCount;       // number of Fourier outputs
  private float growthBase;      // exponential frequency growth
  private NumpyArray lastInput;   // stored for backward cropping
  
  /**
   * @param fourierCount number of Fourier transform output values
   * @param growthBase   exponential growth base for frequencies (e.g. 2.0)
   */
  public FourierTransform(int fourierCount, float growthBase) {
    this(fourierCount, growthBase, null);
  }
  
  /**
   * @param fourierCount number of Fourier transform output values
   * @param growthBase   exponential growth base for frequencies (e.g. 2.0)
   */
  public FourierTransform(int fourierCount, float growthBase, NumpyArray input) {
    if (fourierCount < 0) throw new IllegalArgumentException("fourierCount must be >= 0");
    if (growthBase <= 0) throw new IllegalArgumentException("growthBase must be > 0");
    this.fourierCount = fourierCount;
    this.growthBase = growthBase;
    this.lastInput = input;
  }
  
  @Override
  public NumpyArray forward(NumpyArray input) {
    if (NeuralNetwork.DEBUG) {
      if (input.depth() != 1)
        throw new IllegalArgumentException("Input must have depth 1 (a vector). dimension=" + input.dimension());
      if (input.cols() != 1)
        throw new IllegalArgumentException("Input must be a column vector (cols==1). dimension=" + input.dimension());
    }
    
    // store input for gradient cropping later
    lastInput = input;
    
    final int inputSize = input.rows();
    final int outSize = inputSize * fourierCount * 2; // sin + cos
    
    float[][][] outData = new float[1][outSize][1];
    float[][][] inData = input.data;
    
    int idx = 0;
    
    for (int n = 0; n < inputSize; n++) {
      double x = inData[0][n][0];
      
      for (int k = 0; k < fourierCount; k++) {
        double freq = Math.pow(growthBase, k);
        double v = freq * x;
        
        outData[0][idx++][0] = (float) Math.sin(v);
        outData[0][idx++][0] = (float) Math.cos(v);
      }
    }
    
    return new NumpyArray(outData);
  }
  
  @Override
  public NumpyArray backward(NumpyArray outputGradient, float learningRate) {
    if (lastInput == null)
      throw new IllegalStateException("Backward called before forward.");
    
    final int inputSize = lastInput.rows();
    final int expectedOut = inputSize * fourierCount * 2;
    
    if (NeuralNetwork.DEBUG) {
      if (outputGradient.rows() != expectedOut)
        throw new IllegalArgumentException(
            "Wrong gradient size: expected " + expectedOut + " got " + outputGradient.rows());
    }
    
//    // Only pass through the original input gradients (first inputSize elements)
//    float[][][] gradIn = new float[1][inputSize][1];
//    float[][][] outGrad = outputGradient.data;
//    
//    for (int i = 0; i < inputSize; i++) {
//      gradIn[0][i][0] = outGrad[0][i][0];
//    }
//    
//    // Ignore Fourier-feature gradients entirely
//    return new NumpyArray(gradIn);

    float[][][] gradIn = new float[1][inputSize][1];
    float[][][] outGrad = outputGradient.data;
    float[][][] inData = lastInput.data;

    int idx = 0;

    for (int n = 0; n < inputSize; n++) {
      double x = inData[0][n][0];
      double dx = 0.0;

      for (int k = 0; k < fourierCount; k++) {
        double freq = Math.pow(growthBase, k);
        double v = freq * x;

        float dSin = outGrad[0][idx++][0];
        float dCos = outGrad[0][idx++][0];

        dx += dSin * Math.cos(v) * freq;      // derivative of sin(freq * x)
        dx += dCos * -Math.sin(v) * freq;     // derivative of cos(freq * x)
      }

      gradIn[0][n][0] = (float) dx;
    }

    return new NumpyArray(gradIn);
  }
  
  
  @Override
  public Layer deepCopy() {
    NumpyArray input = this.lastInput == null ? null : this.lastInput.copy();
    return new FourierTransform(fourierCount, growthBase, input);
  }
  
  @Override
  public String toString() {
    return "FourierTransformLayer(pass-through + " + fourierCount + " Fourier outputs, base=" + growthBase + ")";
  }
  
  public int[] outputDimension(int inputDimension) {
    return new int[]{inputDimension * fourierCount * 2};
  }
}