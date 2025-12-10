package models;

import models.dataset.MNISTLoader;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.lossfunction.BinaryCrossEntropy;
import neuralnetwork.lossfunction.LossFunction;
import neuralnetwork.lossfunction.SoftmaxCrossEntropy;
import neuralnetwork.math.Tensor;
import oldneuralnetwork.layer.activation.SoftMax;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.concurrent.ThreadLocalRandom;

public class GAN extends JFrame {
    public static void main(String[] args) {
        new GAN();
    }
    
    Scene scene;
    
    public GAN() {
        setTitle("GAN");
        
        scene = new Scene();
        add(scene);
        
        setSize(600, 300);
        setVisible(true);
        setLocationRelativeTo(null);
        
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        
        startAsyncThreads();
    }
    
    void startAsyncThreads() {
        new Thread(() -> {
            while (true) {
                scene.think();
                try {
                    Thread.sleep(1500);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }).start();
        
        new Thread(() -> {
            while (true) {
                scene.train();
            }
        }).start();
        
        new Thread(() -> {
            while (true) {
                scene.repaint();
                try {
                    Thread.sleep(16);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }).start();
    }
    
    class Scene extends JPanel {
        NeuralNetwork discriminator;
        NeuralNetwork generator;
        int imageResolution = 28;
        int[][] images;
        int[] labels;
        int noiseCount = 10;
        //    LossFunction generatorLoss = new AbsoluteLoss();
        LossFunction discriminatorLoss = new BinaryCrossEntropy();
        float learningRate = 0.0001f;
        
        public Scene() {
            images = MNISTLoader.readTrainImagesSafe().stream().toArray(int[][]::new);
            if (imageResolution != 28)
                for (int i = 0; i < images.length; i++)
                    images[i] = downScale(images[i], imageResolution);
            labels = MNISTLoader.readTrainLabelsSafe().stream().mapToInt(i -> i).toArray();
            
            generator = new NetworkBuilder()
                .optimizer.adam()
                .distribution.xavier()
                .layer.dense(noiseCount + 10, 50)
                .activation.sigmoid()
                .layer.dense(50, 150)
                .activation.sigmoid()
                .layer.dense(150, 200)
                .activation.sigmoid()
                .layer.dense(200, 200)
                .activation.sigmoid()
                .layer.dense(200, imageResolution * imageResolution)
                .activation.sigmoid()
                .build();
            
            discriminator = new NetworkBuilder()
                .optimizer.adam()
                .distribution.xavier()
                .layer.dense(imageResolution * imageResolution + 10, 200)
                .activation.sigmoid()
                .layer.dense(200, 100)
                .activation.sigmoid()
                .layer.dense(100, 50)
                .activation.sigmoid()
                .layer.dense(50, 1)
                .activation.sigmoid()
                .build();
        }
        
        
        float createNoise() {
//            return (float) Math.random() * Float.MAX_VALUE - Float.MAX_VALUE / 2;
            return ThreadLocalRandom.current().nextFloat(Float.MIN_VALUE, Float.MAX_VALUE);
        }
        
        int trainIndex;
        
        public void train() {
            trainIndex++;
            if (trainIndex >= images.length)
                trainIndex = 0;
            
            float[] noiseData = new float[noiseCount + 10];
            for (int i = 0; i < noiseCount; i++)
                noiseData[i] = createNoise();
            
            int fakeLabel = (int) (Math.random() * 10f);
            noiseData[noiseCount + fakeLabel] = 1;
            
            float[] generatedData = generator.predict(Tensor.of(noiseData)).transpose().toFlatArray();
            float[] discriminatorInput = new float[generatedData.length + 10];
            System.arraycopy(generatedData, 0, discriminatorInput, 0, generatedData.length);
            discriminatorInput[generatedData.length + fakeLabel] = 1;
//      float discriminated = discriminator.predict(Tensor.of(discriminatorInput)).toFlatArray()[0];
            
            // train generator
            Tensor grad = discriminator.trainSingleGrad(
                discriminatorLoss,
                Tensor.of(discriminatorInput),
                Tensor.of(new float[] { 0 }),
                learningRate
            );
            float[] data = grad.transpose().toFlatArray();
            float[] truncated = new float[imageResolution * imageResolution];
            System.arraycopy(data, 0, truncated, 0, truncated.length);
            grad = Tensor.of(truncated);
            
            generator.backpropagate(
                grad,
                learningRate
            );
            
            // train discriminator
            int[] pixels = images[trainIndex];
            int label = labels[trainIndex];
            float[] realData = pixelsToFloat(pixels);
//      discriminatorInput = new float[realData.length + 10];
            System.arraycopy(realData, 0, discriminatorInput, 0, realData.length);
            discriminatorInput[realData.length + label] = 1;
            
            discriminator.trainSingle(
                discriminatorLoss,
                Tensor.of(discriminatorInput),
                Tensor.of(new float[] { 0 }),
                learningRate * 0.01f
            );

//      discriminatorInput = new float[generatedData.length + 10];
            System.arraycopy(generatedData, 0, discriminatorInput, 0, generatedData.length);
            discriminatorInput[generatedData.length + fakeLabel] = 1;
            discriminator.trainSingle(
                discriminatorLoss,
                Tensor.of(discriminatorInput),
                Tensor.of(new float[] { 1 }),
                learningRate * 0.01f
            );
        }
        
        int thinkIndex;
        int thinkLabel;
        float[] noiseThink = new float[noiseCount];
        
        void think() {
            thinkIndex++;
            thinkLabel++;
            
            if (thinkLabel > 9)
                thinkLabel = 0;
            thinkIndex %= images.length;
            
            for (int i = 0; i < noiseCount; i++) {
                noiseThink[i] = createNoise();
            }
        }
        
        public void paint(Graphics graphics) {
            if (thinkIndex == 0)
                return;
            
            float[] thinkInput = new float[noiseCount + 10];
            System.arraycopy(noiseThink, 0, thinkInput, 0, noiseThink.length);
            thinkInput[noiseCount + thinkLabel] = 1;
            float[] generatedData = generator.predictThreadSafe(thinkInput).transpose().toFlatArray();
            int[] pixels = floatToPixels(generatedData);
            
            BufferedImage bufferedImage = new BufferedImage(imageResolution, imageResolution, BufferedImage.TYPE_INT_RGB);
            
            for (int i = 0; i < imageResolution * imageResolution; i++) {
                int brightness = pixels[i];
                brightness = Math.max(Math.min(brightness, 255), 0);
                bufferedImage.setRGB(
                    i / imageResolution,
                    i % imageResolution,
                    new Color(brightness, brightness, brightness).getRGB()
                );
            }
            
            graphics.drawImage(
                bufferedImage,
                0, 0,
                getWidth() / 2, getHeight(),
                null
            );
        }
        
        static int[] downScale(int[] pixels, int to) {
            int[] resized = new int[to * to];
            
            for (int i = 0; i < pixels.length; i++) {
                int x = i % 28;
                int y = i / 28;
                
                x /= (float) 28 / to;
                y /= (float) 28 / to;
                int i2 = x + y * to;
                
                resized[i2] += pixels[i];
            }
            
            float divide = (float) pixels.length / resized.length;
            for (int i = 0; i < resized.length; i++) {
                resized[i] = (int) (resized[i] / divide);
                resized[i] = Math.min(Math.max(resized[i], 0), 255);
            }
            
            return resized;
        }
        
        static float[] pixelsToFloat(int[] pixels) {
            float[] output = new float[pixels.length];
            for (int i = 0; i < pixels.length; i++) {
                output[i] = pixels[i] / 255f;
            }
            return output;
        }
        
        static int[] floatToPixels(float[] pixels) {
            int[] output = new int[pixels.length];
            for (int i = 0; i < pixels.length; i++) {
                output[i] = (int) (pixels[i] * 255d);
            }
            return output;
        }
    }
}