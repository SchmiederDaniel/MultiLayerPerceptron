package neuralnetwork.layer;

import neuralnetwork.init.Distribution;
import neuralnetwork.math.*;
import neuralnetwork.optimizer.*;

public class Conv2D extends LearnableLayer {
    private final int numFilters;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final OptimizerType optimizerType;
    
    // Cache for backprop
    private Matrix lastInputCols;
    private int[] lastInputShape;
    private Optimizer optimizer;
    
    public Conv2D(int inputDepth, int numFilters, int kernelSize, int stride, int padding,
                  Distribution distribution, OptimizerType optimizerType) {
        super(null, null);
        this.numFilters = numFilters;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.optimizerType = optimizerType;
        
        int receptiveField = inputDepth * kernelSize * kernelSize;
        
        // Init W: (Rows=Filters, Cols=ReceptiveField)
        this.W = initRandom(receptiveField, numFilters, distribution);
        this.b = new Vector(new float[numFilters]);
        this.optimizer = createOptimizer(optimizerType);
    }
    
    // Private constructor for cloning
    private Conv2D(Matrix W, Vector b, int numFilters, int k, int s, int p, OptimizerType optType) {
        super(null, null);
        this.W = W;
        this.b = b;
        this.numFilters = numFilters;
        this.kernelSize = k;
        this.stride = s;
        this.padding = p;
        this.optimizerType = optType;
        this.optimizer = createOptimizer(optType);
    }
    
    private Optimizer createOptimizer(OptimizerType type) {
        return switch (type) {
            case SGD -> new SGD(this);
            case Adam -> new Adam(this);
        };
    }
    
    private static Matrix initRandom(int inSize, int outSize, Distribution dist) {
        dist.setInputSize(inSize);
        dist.setOutputSize(outSize);
        float[][] w = new float[outSize][inSize];
        for (int i = 0; i < outSize; i++)
            for (int j = 0; j < inSize; j++)
                w[i][j] = dist.randomWeight();
        return new Matrix(w);
    }
    
    @Override
    public Tensor forward(Tensor input) {
        if (input.rank() != 3) throw new IllegalArgumentException("Conv2D requires 3D input. Got rank " + input.rank() + " instead.");
        
        Tensor3D inTensor = (Tensor3D) input;
        this.lastInputShape = inTensor.shape();
        int d = lastInputShape[0], h = lastInputShape[1], w = lastInputShape[2];
        
        int outH = (h + 2 * padding - kernelSize) / stride + 1;
        int outW = (w + 2 * padding - kernelSize) / stride + 1;
        
        // 1. Im2Col
        this.lastInputCols = im2col(inTensor, d, h, w, outH, outW);
        
        // 2. Linear projection: W @ Cols
        Matrix outCols = (Matrix) W.matmul(lastInputCols);
        
        // 3. Add Bias (Manual loop required: Matrix.add(Vector) broadcasts to rows, but we need cols)
        addBias(outCols, b);
        
        // 4. Reshape result directly to Tensor3D using your base class logic
        return outCols.reshape(numFilters, outH, outW);
    }
    
    @Override
    public Tensor backward(Tensor outputGradient, float learningRate) {
        Tensor3D gradOut = (Tensor3D) outputGradient;
        int outH = gradOut.shape()[1];
        int outW = gradOut.shape()[2];
        int pixels = outH * outW;
        
        // 1. Reshape gradient to Matrix [filters, pixels]
        Matrix gradCols = (Matrix) gradOut.reshape(numFilters, pixels);
        
        // 2. Gradients w.r.t Parameters
        Matrix dW = (Matrix) gradCols.matmul(lastInputCols.transpose());
        
        // db is sum of rows (Matrix doesn't have rowSum, so manual loop)
        float[] dbVals = new float[numFilters];
        float[][] gData = gradCols.getValues();
        for (int i = 0; i < numFilters; i++) {
            float sum = 0;
            for (float val : gData[i]) sum += val;
            dbVals[i] = sum;
        }
        
        // 3. Gradient w.r.t Input (dX)
        Matrix dXCols = (Matrix) W.transpose().matmul(gradCols);
        
        // 4. Col2Im
        Tensor3D dInput = col2im(dXCols, lastInputShape, outH, outW);
        
        // 5. Update
        if (optimizer != null) {
            optimizer.step(dW.getValues(), dbVals, learningRate);
        } else {
            // Fallback
            W = (Matrix) W.subtract(new Matrix(dW.getValues()).multiply(Tensor.of(learningRate)));
            b = (Vector) b.subtract(new Vector(dbVals).multiply(Tensor.of(learningRate)));
        }
        
        return dInput;
    }
    
    private void addBias(Matrix m, Vector b) {
        float[][] data = m.getValues();
        float[] bVal = b.getValues();
        for (int i = 0; i < data.length; i++) {
            float bias = bVal[i];
            for (int j = 0; j < data[0].length; j++) {
                data[i][j] += bias;
            }
        }
    }
    
    // --- Core Convolution Logic (Unavoidable Complexity) ---
    
    private Matrix im2col(Tensor3D img, int d, int h, int w, int outH, int outW) {
        float[][][] data = img.getValues();
        int k = kernelSize;
        float[][] colData = new float[d * k * k][outH * outW];
        
        int col = 0;
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                int startY = y * stride - padding;
                int startX = x * stride - padding;
                int row = 0;
                
                for (int c = 0; c < d; c++) {
                    for (int ky = 0; ky < k; ky++) {
                        for (int kx = 0; kx < k; kx++) {
                            int iy = startY + ky;
                            int ix = startX + kx;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                colData[row][col] = data[c][iy][ix];
                            }
                            row++;
                        }
                    }
                }
                col++;
            }
        }
        return new Matrix(colData);
    }
    
    private Tensor3D col2im(Matrix cols, int[] shape, int outH, int outW) {
        int d = shape[0], h = shape[1], w = shape[2];
        float[][][] img = new float[d][h][w];
        float[][] cData = cols.getValues();
        int k = kernelSize;
        
        int col = 0;
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                int startY = y * stride - padding;
                int startX = x * stride - padding;
                int row = 0;
                
                for (int c = 0; c < d; c++) {
                    for (int ky = 0; ky < k; ky++) {
                        for (int kx = 0; kx < k; kx++) {
                            int iy = startY + ky;
                            int ix = startX + kx;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                img[c][iy][ix] += cData[row][col];
                            }
                            row++;
                        }
                    }
                }
                col++;
            }
        }
        return new Tensor3D(img);
    }
    
    @Override
    public Conv2D clone() {
        return new Conv2D((Matrix) W.deepCopy(), (Vector) b.deepCopy(),
            numFilters, kernelSize, stride, padding, optimizerType);
    }
}