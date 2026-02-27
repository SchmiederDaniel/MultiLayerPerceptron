package models.dataset;

import neuralnetwork.math.Tensor;
import org.jetbrains.annotations.NotNull;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public abstract class DataSet implements Iterable<Batch> {
    private final int batchSize;
    private final int seed;
    
    protected DataSet(int batchSize, int seed) {
        this.seed = seed;
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0");
        }
        this.batchSize = batchSize;
    }
    
    protected abstract Tensor[] inputs();
    
    protected abstract Tensor[] targets();
    
    
    @Override
    @NotNull
    public Iterator<Batch> iterator() {
        Tensor[] allInputs = inputs();
        Tensor[] allTargets = targets();
        shuffleCoupledArrays(allInputs, allTargets, seed);
        
        if (allInputs.length != allTargets.length) {
            throw new IllegalStateException("Inputs and targets must have same length");
        }
        
        return new Iterator<>() {
            private int index = 0;
            
            @Override
            public boolean hasNext() {
                return index < allInputs.length;
            }
            
            @Override
            public Batch next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                
                int end = Math.min(index + batchSize, allInputs.length);
                
                Tensor[] batchInputs =
                    Arrays.copyOfRange(allInputs, index, end);
                Tensor[] batchTargets =
                    Arrays.copyOfRange(allTargets, index, end);
                
                index = end;
                
                return new Batch(batchInputs, batchTargets);
            }
        };
    }
    
    /**
     * Shuffles two arrays simultaneously, maintaining the index relationship
     * between elements of the same index.
     *
     * @param <T>    The type of the first array (e.g., Features)
     * @param <U>    The type of the second array (e.g., Labels)
     * @param array1 The first array to shuffle
     * @param array2 The second array to shuffle
     * @throws IllegalArgumentException if arrays are null or have different lengths
     */
    public static <T, U> void shuffleCoupledArrays(T[] array1, U[] array2, int seed) {
        // 1. Basic Validation
        if (array1 == null || array2 == null) {
            throw new IllegalArgumentException("Arrays cannot be null.");
        }
        if (array1.length != array2.length) {
            throw new IllegalArgumentException("Arrays must be of the same length.");
        }
        
        // 2. Use ThreadLocalRandom for better performance in multithreaded envs
        Random rnd = new Random(seed);
        
        // 3. Fisher-Yates Shuffle Algorithm
        for (int i = array1.length - 1; i > 0; i--) {
            // Generate a random index between 0 and i (inclusive)
            int index = rnd.nextInt(i + 1);
            
            // Swap elements in Array 1
            T temp1 = array1[index];
            array1[index] = array1[i];
            array1[i] = temp1;
            
            // Swap elements in Array 2 (using the SAME index)
            U temp2 = array2[index];
            array2[index] = array2[i];
            array2[i] = temp2;
        }
    }
}