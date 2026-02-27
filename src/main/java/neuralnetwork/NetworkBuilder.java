package neuralnetwork;

import neuralnetwork.activation.*;
import neuralnetwork.init.CustomDistribution;
import neuralnetwork.init.Distribution;
import neuralnetwork.init.NormalDistribution;
import neuralnetwork.init.Xavier;
import neuralnetwork.layer.*;
import neuralnetwork.optimizer.OptimizerType;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
    List<Layer> layerList = new ArrayList<>();
    public Activation activation = new Activation();
    public Layers layer = new Layers();
    public PrivateDistribution distribution = new PrivateDistribution();
    private OptimizerType optimizerType = OptimizerType.SGD;
    
    private NetworkBuilder networkBuilder = this;
    private Distribution savedDistribution = new NormalDistribution();
    public Optimizer optimizer = new Optimizer();
    
    public class Optimizer {
        public NetworkBuilder sgd() {
            optimizerType = OptimizerType.SGD;
            return networkBuilder;
        }
        
        public NetworkBuilder adam() {
            optimizerType = OptimizerType.Adam;
            return networkBuilder;
        }
    }
    
    public class PrivateDistribution {
        public NetworkBuilder normalDistribution() {
            savedDistribution = new NormalDistribution();
            return networkBuilder;
        }
        
        public NetworkBuilder zeroBias() {
            savedDistribution = new CustomDistribution(0, 0.0001f, -1, 1);
            return networkBuilder;
        }
        
        public NetworkBuilder ownBias(float from, float to) {
            savedDistribution = new CustomDistribution(from, to, -1, 1);
            return networkBuilder;
        }
        
        public NetworkBuilder customDistribution(float fromBias, float toBias, float fromWeight, float toWeight) {
            savedDistribution = new CustomDistribution(fromBias, toBias, fromWeight, toWeight);
            return networkBuilder;
        }
        
        public NetworkBuilder xavier() {
            savedDistribution = new Xavier(true);
            return networkBuilder;
        }
        
        public NetworkBuilder xavier(boolean normal) {
            savedDistribution = new Xavier(normal);
            return networkBuilder;
        }
    }
    
    public class Activation {
//    public NetworkBuilder reLU() {
//      layerList.add(new ReLU());
//      return networkBuilder;
//    }
        
        public NetworkBuilder sigmoid() {
            layerList.add(new Sigmoid());
            return networkBuilder;
        }
        
        //    public NetworkBuilder softMax() {
//      layerList.add(new SoftMax());
//      return networkBuilder;
//    }
//    
        public NetworkBuilder tanh() {
            layerList.add(new Tanh());
            return networkBuilder;
        }
        
        //    
//    public NetworkBuilder elu() {
//      layerList.add(new ELU());
//      return networkBuilder;
//    }
//    
        public NetworkBuilder leakyReLu() {
            layerList.add(new LeakyReLU());
            return networkBuilder;
        }
        
        public NetworkBuilder gelu() {
            layerList.add(new GELU());
            return networkBuilder;
        }
        
        
        public NetworkBuilder softMax() {
            layerList.add(new Softmax());
            return networkBuilder;
        }
    }
    
    public class Layers {
        public NetworkBuilder dense(int input_size, int output_size) {
            layerList.add(new Dense(input_size, output_size, savedDistribution, optimizerType));
            return networkBuilder;
        }
        
        public NetworkBuilder conv2D(int inputDepth, int numFilters, int kernelSize, int stride, int padding) {
            layerList.add(new Conv2D(inputDepth, numFilters, kernelSize, stride, padding, savedDistribution, optimizerType));
            return networkBuilder;
        }
        
        public NetworkBuilder flatten() {
            layerList.add(new Flatten());
            return networkBuilder;
        }
        
        public NetworkBuilder reshape(int... shape) {
            layerList.add(new Reshape(shape));
            return networkBuilder;
        }
        
        //    public NetworkBuilder dense(NumpyArray weights, NumpyArray bias) {
//      layerList.add(new Dense(weights, bias));
//      return networkBuilder;
//    }
//    
        public NetworkBuilder dropOut(float dropOutRate) {
            layerList.add(new Dropout(dropOutRate));
            return networkBuilder;
        }
        
        public NetworkBuilder averageNorm() {
            layerList.add(new AverageNorm());
            return networkBuilder;
        }

//    public NetworkBuilder dropOut(double dropOutRate) {
//      return dropOut((float) dropOutRate);
//    }
    }
    
    public NeuralNetwork build() {
        return new NeuralNetwork(layerList.toArray(new Layer[0]));
    }
    
    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        
        for (Layer layer : layerList) {
            stringBuilder.append(layer);
            stringBuilder.append(", ");
        }
        
        String outputString;
        if (layerList.size() == 0) {
            stringBuilder.append("Empty");
            outputString = stringBuilder.toString();
        } else {
            outputString = stringBuilder.substring(0, stringBuilder.length() - 2);
        }
        
        return outputString;
    }
}