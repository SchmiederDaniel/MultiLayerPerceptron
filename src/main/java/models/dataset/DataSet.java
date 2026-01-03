package models.dataset;

import neuralnetwork.math.Tensor;
import org.jetbrains.annotations.NotNull;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Arrays;

public abstract class DataSet implements Iterable<Batch> {
    
    private final int batchSize;
    
    protected DataSet(int batchSize) {
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
}