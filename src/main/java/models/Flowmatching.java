package models;

import models.dataset.Batch;
import models.dataset.DataSet;
import models.dataset.MNISTLoader;
import models.dataset.MnistDataset;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.lossfunction.LossFunction;
import neuralnetwork.lossfunction.MeanSquareError;
import neuralnetwork.math.Tensor;
import oldneuralnetwork.NumpyArray;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.util.Random;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ThreadLocalRandom;

public class Flowmatching extends JFrame {
    public static void main(String[] args) {
        new Flowmatching();
    }
    Scene scene;
    static JSlider sliderViewSteps = new JSlider(1, Scene.MAX_STEP_SIZE - 1, Scene.MAX_STEP_SIZE / 2);
    JSlider sliderLearningRate = new JSlider(0, 1000000, (int) (Scene.learningRate * 1000000d * 4d));
    
    public Flowmatching() {
        setLayout(null);
        scene = new Scene();
        add(scene);
        setBackground(new Color(120, 120, 120));
        
        JLabel labelNoise = new JLabel("Strength of noising before denoising (0.5):");
        add(labelNoise);
        add(sliderViewSteps);
        sliderViewSteps.addChangeListener(e -> {
            labelNoise.setText("Strength of noising before denoising (" + Math.round((float) sliderViewSteps.getValue() / Scene.MAX_STEP_SIZE * 100d) / 100d + "):");
        });
        
        JLabel labelLR = new JLabel("LearningRate: " + Math.round(Scene.learningRate * 100d) / 100d);
        add(labelLR);
        add(sliderLearningRate);
        sliderLearningRate.addChangeListener(e -> {
            Scene.learningRate = (float) sliderLearningRate.getValue() / sliderLearningRate.getMaximum() / 4f / 10f;
            labelLR.setText("LearningRate: " + String.format("%.7f", Scene.learningRate));
        });
        
        addComponentListener(new ComponentAdapter() {
            public void componentResized(ComponentEvent e) {
                labelNoise.setSize(getWidth(), 20);
                labelNoise.setLocation(10, getHeight() - 150);
                
                sliderViewSteps.setSize(getWidth() - 20, 30);
                sliderViewSteps.setLocation(0, labelNoise.getY() + labelNoise.getHeight());
                
                labelLR.setSize(getWidth(), 20);
                labelLR.setLocation(10, sliderViewSteps.getY() + sliderViewSteps.getHeight());
                
                sliderLearningRate.setSize(getWidth() - 20, 30);
                sliderLearningRate.setLocation(0, labelLR.getY() + labelLR.getHeight());
                
                scene.setSize(getWidth() - 4, getWidth());
            }
        });
        
        setSize(800, 950);
        setVisible(true);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }
    
    class Scene extends JPanel {
        final static int imageResolution = 28;
        int[][] images;
        static int[][] testImages;
        int[] labels;
        int[] testLabels;
        
        public Scene() {
            images = MNISTLoader.trainData().stream().toArray(int[][]::new);
            labels = MNISTLoader.trainLabels().stream().mapToInt(i -> i).toArray();
            
            testImages = MNISTLoader.testData().stream().toArray(int[][]::new);
            testLabels = MNISTLoader.testLabels().stream().mapToInt(i -> i).toArray();
            
            startAsyncThreads();
        }
        
        void startAsyncThreads() {
            new Thread(() -> {
                while (true) {
                    this.bufferedImage = think(Math.max(getWidth(), 1), Math.max(getHeight(), 1));
                }
            }).start();
            
            new Thread(() -> {
                while (true) {
                    thinkIndex++;
                    thinkIndex %= testImages.length;
                    try {
                        Thread.sleep(1800);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                }
            }).start();
            
            new Thread(() -> {
                while (true) {
                    train();
                }
            }).start();
            
            new Thread(() -> {
                while (true) {
                    repaint();
                    try {
                        Thread.sleep(32);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                }
            }).start();
        }
        
        ConcurrentLinkedDeque<Float> lastLosses = new ConcurrentLinkedDeque<>();
        int trainIndex;
        private final int BATCH_SIZE = 32;
        final static int MAX_STEP_SIZE = 50;
        static float learningRate = 0.0001f;
        LossFunction lossFunction = new MeanSquareError();
        static NeuralNetwork neuralNetwork = new NetworkBuilder()
            .optimizer.adam()
            .distribution.xavier()
            .layer.dense(imageResolution * imageResolution, 512)
            .activation.gelu()
            .layer.dense(512, 512)
            .activation.gelu()
            .layer.dense(512,  512)
            .activation.gelu()
            .layer.dense(512, 512)
            .activation.gelu()
            .layer.dense(512, imageResolution * imageResolution)
            .activation.sigmoid()
            .build();
        
        public void train() {
            float lossSum = 0f;
            
            for (Batch batch : new MnistDataset(32, true, trainIndex)) {
                Tensor[] input = batch.inputs;
                Tensor[] targets = batch.targets;
                
                neuralNetwork.train(input, targets, learningRate);
            }
            
            // Average loss and add to deque
            lastLosses.add(lossSum / BATCH_SIZE);
            trainIndex++;
        }
        
        static float[] addNoise(int seed, float[] array, float strength) {
            float[] output = new float[array.length];
            Random random = new Random(seed);
            
            for (int i = 0; i < array.length; i++) {
                float value = array[i];
                float diffrence = random.nextFloat() - value;
                output[i] = value + diffrence * strength;
            }
            return output;
        }
        
        static float convertNoiseStrength(float x) {
            return x - x * x + x;
//      return x * x;
//      return x;
        }
        
        static int thinkIndex;
        
        BufferedImage bufferedImage = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
        
        static BufferedImage think(int width, int height) {
            BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            Graphics2D graphics = bufferedImage.createGraphics();
            
            float[] pixels = pixelsTofloat(testImages[thinkIndex]);
            float[] noisy = addNoise(thinkIndex, pixels, (float) sliderViewSteps.getValue() / MAX_STEP_SIZE);
            
            int[] decodedPixels = floatToPixels(noisy);
            drawPixels(graphics, decodedPixels, 0, height / 5 * 4, width / 5, height / 5);
            decodedPixels = floatToPixels(pixels);
            drawPixels(graphics, decodedPixels, height / 5, height / 5 * 4, width / 5, height / 5);
            
//            
//            drawPixels(graphics, floatToPixels(noisy), height / 5 * 2, height / 5 * 4, width / 5, height / 5);
//            
//            float[] difference = NumpyArray.of(noisyOriginal).subtract(NumpyArray.of(noisy)).add(NumpyArray.of(0.5f)).transpose().data[0];
//            drawPixels(graphics, floatToPixels(difference), height / 5 * 3, height / 5 * 4, width / 5, height / 5);
            
            return bufferedImage;
        }
        
        public void paint(Graphics graphics) {
            super.paint(graphics);
            if (thinkIndex == 0)
                return;

            setTitle("training steps: " + trainIndex + " learning rate: " + String.format("%.8f", learningRate) + " diffusion step size: " + MAX_STEP_SIZE);
            graphics.drawImage(bufferedImage, 0, 0, getWidth(), getHeight(), null);
            graphics.setColor(Color.white);
            float lossAVG = lastLosses.stream().reduce(0f, Float::sum) / lastLosses.size();
            graphics.drawString("Loss: " + String.format("%.8f", lossAVG), getWidth() - 120, getHeight() - 20);
            while (lastLosses.size() > 500)
                lastLosses.removeFirst();
        }
        
        static float[] addValuesToArray(float[] array, float... values) {
            float[] output = new float[array.length + values.length];
            System.arraycopy(array, 0, output, 0, array.length);
            System.arraycopy(values, 0, output, array.length, values.length);
            return output;
        }
        
        static void drawPixels(Graphics graphics, int[] pixels, int x, int y, int width, int height) {
            BufferedImage bufferedImage = new BufferedImage(imageResolution, imageResolution, BufferedImage.TYPE_INT_RGB);
            
            for (int i = 0; i < imageResolution * imageResolution; i++) {
                int brightness = Math.max(Math.min(pixels[i], 255), 0);
                bufferedImage.setRGB(
                    i % imageResolution,
                    i / imageResolution,
                    new Color(brightness, brightness, brightness).getRGB()
                );
            }
            
            graphics.drawImage(
                bufferedImage,
                x, y,
                width, height,
                null
            );
        }
        
        static float[] pixelsTofloat(int[] pixels) {
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
        
        static float[] minmax(float[] array, float min, float max) {
            float[] output = new float[array.length];
            for (int i = 0; i < array.length; i++) {
                float value = array[i];
                output[i] = value < min ? min : value;
                output[i] = value > max ? max : value;
            }
            return output;
        }
    }
}