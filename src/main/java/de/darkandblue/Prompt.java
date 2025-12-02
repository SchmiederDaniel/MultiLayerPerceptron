package de.darkandblue;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.lossfunction.AbsoluteLoss;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;

public class Prompt {
//    NeuralNetwork neuralNetworkEncoder = new NetworkBuilder()
//            .layer.dense(784, 80)
//            .activation.sigmoid()
//            .layer.dense(80, 80)
//            .activation.sigmoid()
//            .layer.dense(80, 1)
//            .activation.sigmoid()
//            .build();
//    NeuralNetwork neuralNetworkDecoder = new NetworkBuilder()
//            .layer.dense(1, 80)
//            .activation.sigmoid()
//            .layer.dense(80, 80)
//            .activation.sigmoid()
//            .layer.dense(80, 784)
//            .activation.sigmoid()
//            .build();
//    private final static LossFunction decoderLossFunction = new AbsoluteLoss();
//    private final static LossFunction encoderLossFunction = new AbsoluteLoss();
//
//
//    int trainIndex;
//
//    void train() {
//        trainIndex++;
//        trainIndex %= 60000;
//
//        int[] pixels = imageList.get(trainIndex);
//        float[] x = pixelsToFloat(pixels);
//
//        NumpyArray encoderOutput = neuralNetworkEncoder.predict(NumpyArray.of(x));
//        NumpyArray decoderOutput = neuralNetworkDecoder.predict(encoderOutput);
//
//        NumpyArray grad = neuralNetworkDecoder.trainWithoutPredict(decoderLossFunction, decoderOutput, x, learningRate);
//
//        // making gradient ascent on the encoder
//        grad = grad.multiply(-1);
//
//        neuralNetworkEncoder.trainWithoutPredict(encoderLossFunction, encoderOutput, grad, learningRate * 0.5f);
//    }
  
}