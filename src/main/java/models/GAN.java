package models;

import models.dataset.MNISTLoader;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.lossfunction.BinaryCrossEntropy;
import neuralnetwork.lossfunction.LossFunction;
import neuralnetwork.math.Tensor;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.concurrent.CopyOnWriteArrayList;
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
        
        setSize(800, 400);
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
        LossFunction lossFunction = new BinaryCrossEntropy();
        LossFunction generatorLossFunction = new BinaryCrossEntropy();
        float learningRate = 0.00001f;
        
        public Scene() {
            images = MNISTLoader.trainData().stream().toArray(int[][]::new);
            if (imageResolution != 28)
                for (int i = 0; i < images.length; i++)
                    images[i] = downScale(images[i], imageResolution);
            labels = MNISTLoader.trainLabels().stream().mapToInt(i -> i).toArray();
            
            generator = new NetworkBuilder()
                .optimizer.adam()
                .distribution.xavier()
                .layer.dense(noiseCount, 128)
                .activation.gelu()
                .layer.dense(128, 256)
                .activation.gelu()
                .layer.dense(256, 512)
                .activation.gelu()
                .layer.dense(512, 1024)
                .activation.gelu()
                .layer.dense(1024, imageResolution * imageResolution)
                .activation.sigmoid()
                .build();
            
            discriminator = new NetworkBuilder()
                .optimizer.adam()
                .distribution.xavier()
                .layer.dense(imageResolution * imageResolution, 512)
                .activation.gelu()
                .layer.dropOut(0.3f)
                .layer.dense(512, 256)
                .activation.gelu()
                .layer.dropOut(0.3f)
                .layer.dense(256, 256)
                .activation.gelu()
                .layer.dense(256, 1)
                .activation.sigmoid()
                .build();
        }
        
        
        float createNoise() {
//            return (float) Math.random() * Float.MAX_VALUE - Float.MAX_VALUE / 2;
//            return ThreadLocalRandom.current().nextFloat(Float.MIN_VALUE, Float.MAX_VALUE);
            return ThreadLocalRandom.current().nextFloat(-1f, 1f);
        }
        
        int trainIndex;
        
        CopyOnWriteArrayList<Float> generatorLoss = new CopyOnWriteArrayList<>();
        CopyOnWriteArrayList<Float> discriminatorLossReal = new CopyOnWriteArrayList<>();
        CopyOnWriteArrayList<Float> discriminatorLossFake = new CopyOnWriteArrayList<>();
        
        public void train() {
            trainIndex++;
            if (trainIndex >= images.length)
                trainIndex = 0;
            
            trainGenerator();
            trainDiscriminator();
        }
        
        Tensor generateData() {
            float[] noiseData = new float[noiseCount];
            for (int i = 0; i < noiseCount; i++)
                noiseData[i] = createNoise();
            
            return generator.predict(Tensor.of(noiseData));
        }
        
        void trainGenerator() {
            Tensor generatedData = generateData();
            float[] generatedFloat = generatedData.toFlatArray();
            
            // train generator
            Tensor grad = discriminator.trainSingleGrad(
                lossFunction,
                Tensor.of(generatedFloat),
                Tensor.of(new float[] { 1 }),
                0
            );
            generatorLoss.add(generatorLossFunction.loss(grad, generatedData));
            generator.backpropagate(
                grad,
                learningRate
            );
        }
        
        boolean trainDiscriminatorLast = false;
        
        void trainDiscriminator() {
            Tensor generatedData = generateData();
            
            if (trainDiscriminatorLast) {
                // train discriminator
                int[] pixels = images[trainIndex];
                float[] realData = pixelsToFloat(pixels);
                discriminatorLossReal.add(
                    discriminator.trainSingle(
                        lossFunction,
                        Tensor.of(realData),
                        Tensor.of(new float[] { 1 }),
                        learningRate
                    )
                );
            } else {
                discriminatorLossFake.add(
                    discriminator.trainSingle(
                        lossFunction,
                        generatedData,
                        Tensor.of(new float[] { 0 }),
                        learningRate
                    )
                );
            }
            trainDiscriminatorLast = !trainDiscriminatorLast;
        }
        
        int thinkIndex;
        float[] noiseThink = new float[noiseCount];
        
        void think() {
            thinkIndex++;
            
            thinkIndex %= images.length;
            
            for (int i = 0; i < noiseCount; i++) {
                noiseThink[i] = createNoise();
            }
        }
        
        private void drawLine(Graphics graphics, CopyOnWriteArrayList<Float> set, Color color, int x, int y, float width, float height) {
            graphics.setColor(color);
            float partialWidth = width / set.size();
            ((Graphics2D) graphics).setStroke(new BasicStroke(1f));
            for (int i = 1; i < set.size(); i++) {
                float value = 1f - set.get(i);
                float last = 1f - set.get(i - 1);
                graphics.drawLine(
                    x + (int) ((i - 1) * partialWidth),
                    y + (int) (height * last),
                    x + (int) (i * partialWidth),
                    y + (int) (height * value)
                );
            }
        }
        
//        float getLearningRate(CopyOnWriteArrayList<Float> list, boolean invert) {
//            if (list.isEmpty())
//                return 0.0001f;
//            Float last = list.getLast();
//            last=Math.min(1, last);
//            if (invert)
//                last = 1 - last;
//            return last * 0.00001f;
//        }
        
        public void paint(Graphics graphics) {
            super.paint(graphics);
            if (thinkIndex == 0)
                return;
            graphics.setFont(new Font("Arial", Font.BOLD, 12));
            
            setTitle("training steps: " + trainIndex + " Losscount: " + generatorLoss.size());
            
            while (discriminatorLossFake.size() > 30000) {
                discriminatorLossFake.removeFirst();
                discriminatorLossReal.removeFirst();
                generatorLoss.removeFirst();
                generatorLoss.removeFirst();
            }
            
            int height = getHeight();
            drawLine(graphics, generatorLoss, new Color(0, 100, 0), getWidth() / 2, 0, height, height);
            drawLine(graphics, discriminatorLossReal, Color.red, getWidth() / 2, 0, height, height);
            drawLine(graphics, discriminatorLossFake, Color.MAGENTA, getWidth() / 2, 0, height, height);
            
            if (!discriminatorLossReal.isEmpty()) {
                graphics.setColor(new Color(60, 0, 0));
                graphics.drawString("Discriminator loss real: " + discriminatorLossReal.getLast(), getWidth() / 2 + 10, 10);
                graphics.setColor(new Color(60, 0, 60));
                graphics.drawString("Discriminator loss fake: " + discriminatorLossFake.getLast(), getWidth() / 2 + 10, 25);
                graphics.setColor(new Color(0, 60, 0));
                graphics.drawString("Generator loss: " + generatorLoss.getLast(), getWidth() / 2 + 10, 40);
            }
            
            float[] generatedData = generateData().toFlatArray();
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