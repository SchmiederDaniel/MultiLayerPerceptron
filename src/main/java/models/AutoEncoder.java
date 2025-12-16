package models;

import models.dataset.MNISTLoader;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.lossfunction.*;
import neuralnetwork.math.Tensor;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.List;

public class AutoEncoder extends JFrame {
    public static void main(String[] args) {
        new AutoEncoder();
    }
    
    float learningRate = 0.0001f;
    
    public AutoEncoder() {
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLayout(null);
        
        Scene scene = new Scene();
        add(scene);
        
        JSlider sliderLearningRate = new JSlider(0, 5000, (int) (50000f * learningRate * 3f));
        sliderLearningRate.addChangeListener(e -> {
            learningRate = sliderLearningRate.getValue() / (float) sliderLearningRate.getMaximum() / 300f;
            updateTitle();
        });
        add(sliderLearningRate);
        
        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                super.componentResized(e);
                scene.setSize(getWidth(), getWidth() / 3);
                sliderLearningRate.setLocation(20, getWidth() / 3 + 20);
                sliderLearningRate.setSize(getWidth() - 40, 120);
            }
        });
        
        setSize(1500, 700);
        setVisible(true);
        setLocationRelativeTo(null);
        
        updateTitle();
    }
    
    void updateTitle() {
        setTitle("learningRate: " + String.format("%.7f", learningRate));
    }
    
    class Scene extends JPanel {
        List<int[]> imageList;
        List<Integer> labelList;
        NeuralNetwork neuralNetworkEncoder = new NetworkBuilder()
            .optimizer.adam()
            .distribution.xavier()
            .layer.reshape(1, 28, 28)
            .layer.conv2D(1, 25, 7, 2, 0)
            .activation.gelu()
            .layer.conv2D(25, 25, 3, 2, 0)
            .activation.gelu()
            .layer.conv2D(25, 25, 3, 2, 0)
            .activation.gelu()
            .layer.flatten()
            .layer.dense(100, 1)
            .activation.gelu()
            .build();
        NeuralNetwork neuralNetworkDecoder = new NetworkBuilder()
            .optimizer.adam()
            .distribution.xavier()
            .layer.dense(1, 300)
            .activation.gelu()
            .layer.dense(300, 400)
            .activation.gelu()
            .layer.dense(400, 400)
            .activation.gelu()
            .layer.dense(400, 784)
            .activation.gelu()
            .build();
        private final static LossFunction decoderLossFunction = new L1();
        
        // TODO: wouldn't it be usefull if the loss function of an image gets determined by how much a number looks like a number? 
        // a algorithm would be usefull which compares the generated image and how it deviates from pixels near by from the original
        
        public Scene() {
            imageList = MNISTLoader.readTrainImagesSafe();
            System.out.println(imageList.size());
            labelList = MNISTLoader.readTrainLabelsSafe();
            
            startAsyncThreads();
        }
        
        public void startAsyncThreads() {
            new Thread(() -> {
                while (true) {
                    try {
                        Thread.sleep(16);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    repaint();
                }
            }).start();
            
            new Thread(() -> {
                while (true) {
                    train();
                }
            }).start();
            
            new Thread(() -> {
                while (true) {
                    test();
                }
            }).start();
        }
        
        float testError;
        
        void test() {
            float sumTotal = 0;
            int iterations = 3000;
            for (int i = 0; i < iterations; i++) {
                int[] pixels = imageList.get(i);
                float[] x = pixelsToFloat(pixels);
                
                Tensor encoderOutput = neuralNetworkEncoder.predictThreadSafe(Tensor.of(x));
                Tensor decoderOutput = neuralNetworkDecoder.predictThreadSafe(encoderOutput);
                
                float[] totalError = decoderOutput.transpose().toFlatArray();
                float sum = 0;
                for (int e = 0; e < totalError.length; e++) {
                    sum += Math.abs(x[e] - totalError[e]);
                }
                sumTotal += sum / totalError.length;
            }
            testError = sumTotal / iterations;
        }
        
        int trainIndex;
        
        void train() {
            trainIndex++;
            trainIndex %= 60000;
            
            int[] pixels = imageList.get(trainIndex);
            float[] x = pixelsToFloat(pixels);
            
            Tensor encoderOutput = neuralNetworkEncoder.predict(Tensor.of(x));
//            Tensor decoderOutput = neuralNetworkDecoder.predict(encoderOutput);
            
            Tensor grad = neuralNetworkDecoder.trainSingleGrad(decoderLossFunction, encoderOutput, Tensor.of(x), learningRate);

//       making gradient ascent on the encoder
//      grad = grad.multiply(-1);
            
            neuralNetworkEncoder.backpropagate(grad, learningRate * 0.5f);
        }
        
        float[] pixelsToFloat(int[] pixels) {
            float[] output = new float[pixels.length];
            for (int i = 0; i < pixels.length; i++) {
                output[i] = pixels[i] / 255f;
            }
            return output;
        }
        
        int[] floatToPixels(float[] pixels) {
            int[] output = new int[pixels.length];
            for (int i = 0; i < pixels.length; i++) {
                output[i] = (int) (pixels[i] * 255d);
            }
            return output;
        }
        
        int thinkIndex;
        final static int IMAGE_CHANGE_TIME = 1300;
        long lastImageChange = System.currentTimeMillis();
        
        public void paint(Graphics graphics) {
            super.paint(graphics);
            if (System.currentTimeMillis() - lastImageChange > IMAGE_CHANGE_TIME) {
                thinkIndex++;
                thinkIndex %= 60000;
                
                lastImageChange = System.currentTimeMillis();
            }
            
            int imageSize = getWidth() / 3;
            
            int[] pixels = imageList.get(thinkIndex);
            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            setPixels(pixels, image);
            graphics.drawImage(image, 0, 0, imageSize, imageSize, null);
            
            float[] inputs = pixelsToFloat(pixels);
            Tensor x = neuralNetworkEncoder.predictThreadSafe(inputs);
            
            int[] lowResPixels = floatToPixels(x.transpose().toFlatArray());
            image = new BufferedImage(7, 7, BufferedImage.TYPE_INT_RGB);
            setPixels(lowResPixels, image);
            graphics.drawImage(image, imageSize, 0, imageSize, imageSize, null);
            
            float[] y = neuralNetworkDecoder.predictThreadSafe(x).transpose().toFlatArray();
            int[] predictedPixels = floatToPixels(y);
            image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            setPixels(predictedPixels, image);
            graphics.drawImage(image, imageSize * 2, 0, imageSize, imageSize, null);
            
            graphics.setColor(Color.white);
            graphics.drawString("error: " + testError, 10, imageSize - 20);
        }
        
        void setPixels(int[] pixels, BufferedImage image) {
            int[] imagePixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
            
            for (int i = 0; i < pixels.length; i++) {
                int pixel = Math.min(Math.max(pixels[i], 0), 255);
                imagePixels[i] = new Color(pixel, pixel, pixel).getRGB();
            }
        }
    }
}