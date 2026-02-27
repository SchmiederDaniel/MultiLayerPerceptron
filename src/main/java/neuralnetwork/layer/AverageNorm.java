package neuralnetwork.layer;

import neuralnetwork.math.Tensor;

public class AverageNorm extends Layer {
    private final float MAX_SUM = Float.MAX_VALUE / 2f;
    Tensor inputSum;
    int count;
    boolean limitReached = false;
    
    public AverageNorm() {
    }
    
    public AverageNorm(Tensor inputSum, int count, boolean limitReached) {
        this.inputSum = inputSum;
        this.count = count;
        this.limitReached = limitReached;
    }
    
    @Override
    public Layer clone() {
        if (inputSum == null) return new AverageNorm();
        return new AverageNorm(this.inputSum.deepCopy(), this.count, this.limitReached);
    }
    
    @Override
    public Tensor forward(Tensor input) {
        if (!limitReached) {
            if (inputSum == null) {
                inputSum = input.deepCopy();
                
            } else {
                float[] flatArray = inputSum.add(input).toFlatArray();
                for (float value : flatArray) {
                    if (value > MAX_SUM) limitReached = true;
                }
            }
            count++;
        }

//        Tensor average = inputSum.divide(Tensor.of(count));
//        Tensor normalized = input.divide(average); // "average" can contain 0 which results in division by 0 errors
//        normalized = normalized.multiply(Tensor.of(2));
//        normalized = normalized.subtract(Tensor.of(1));
//        System.out.println(Arrays.toString(normalized.toFlatArray()));
        
        if (count > 1000) {
            Tensor normalized;
            // 1. Define a tiny epsilon to prevent division by zero
            double epsilon = 1e-8;
            // 2. Instead of direct division, add epsilon to the count
            // This handles the case where count might be 0
            Tensor safeCount = Tensor.of(count + epsilon);
            Tensor average = inputSum.divide(safeCount);
            // 3. For the normalized step, calculate the reciprocal of (average + epsilon)
            // This turns the operation into multiplication
            Tensor safeAverageInverse = average.add(Tensor.of(epsilon)).reciprocal();
            normalized = input.multiply(safeAverageInverse);
            // 4. Continue with your linear transformation
            normalized = normalized.multiply(Tensor.of(2)).subtract(Tensor.of(1));
            return normalized;
        }else {
            return input;
        }
    }
    
    @Override
    public Tensor backward(Tensor outputGradient, float learningRate) {
        // 1. Calculate the current average (mu)
        // We use the same logic as forward to ensure consistency
        Tensor average = inputSum.divide(Tensor.of(count));
        
        // 2. The derivative of (2 * (x / average) - 1) with respect to x is (2 / average)
        Tensor derivative = Tensor.of(2).divide(average);
        
        // 3. Chain rule: inputGradient = outputGradient * derivative
        return outputGradient.multiply(derivative);
    }
    
    @Override
    public String toString() {
        return "AverageNorm";
    }
}