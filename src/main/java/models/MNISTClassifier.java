package models;

import models.dataset.MNISTLoader;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.layer.Dense;
import neuralnetwork.layer.Layer;
import neuralnetwork.lossfunction.*;
import neuralnetwork.math.Matrix;
import neuralnetwork.math.Tensor;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MNISTClassifier extends JFrame {
    public static void main(String[] args) throws IOException {
        new MNISTClassifier();
    }
    
    public MNISTClassifier() {
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(900, 500);
        add(new Scene());
        
        setVisible(true);
        setLocationRelativeTo(null);
    }
    
    class Scene extends JPanel {
        LossFunction lossFunction = new BinaryCrossEntropy();
        NeuralNetwork neuralNetwork = new NetworkBuilder()
            .optimizer.adam()
            .layer.dense(28 * 28, 400)
            .activation.sigmoid()
            .layer.dense(400, 200)
            .activation.sigmoid()
            .layer.dense(200, 100)
            .activation.sigmoid()
            .layer.dense(100, 80)
            .activation.sigmoid()
            .layer.dense(80, 10)
            .activation.sigmoid()
            .build();
        List<int[]> imageList = MNISTLoader.readTrainImagesSafe();
        List<Integer> labelList = MNISTLoader.readTrainLabelsSafe();
        List<int[]> testImageList = MNISTLoader.readTestImagesSafe();
        List<Integer> testLabelList = MNISTLoader.readTestLabelsSafe();
        float learningRate = 0.0001f;
        
        public Scene() {
            displayImagePixels = testImageList.get(paintIndex);
            
            startAsyncThreads();
        }
        
        void startAsyncThreads() {
            // generating new display image
            new Thread(() -> {
                while (true) {
                    try {
                        Thread.sleep(2500);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    paintIndex++;
                    paintIndex %= testImageList.size();
                    displayImagePixels = testImageList.get(paintIndex);
                }
            }).start();
            
            // rendering
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
            
            // testing
            new Thread(() -> {
                while (true) {
                    test();
                }
            }).start();
            
            // training
            new Thread(() -> {
                while (true) {
                    train();
                }
            }).start();
        }
        
        int trainIndex;
        
        void train() {
            trainIndex++;
            trainIndex %= imageList.size();
            
            int[] pixels = imageList.get(trainIndex);
            int label = labelList.get(trainIndex);
            
            float[] x = pixelsToFloat(pixels);
            float[] y = new float[10];
            y[label] = 1;
            
            neuralNetwork.trainSingle(lossFunction, Tensor.of(x), Tensor.of(y), learningRate);
            
            for (Layer layer : neuralNetwork.layers) {
                if (layer instanceof Dense dense) {
                    if (trainIndex % 1000 == 0) { // remove faded out graidents
                        Matrix w = dense.W;
                        float[][] values = w.getValues();
                        for (int rowIndex = 0; rowIndex < values.length; rowIndex++) {
                            float[] row = values[rowIndex];
                            for (int colIndex = 0; colIndex < values[0].length; colIndex++) {
                                float v = row[colIndex];
                                values[rowIndex][colIndex] = v;
                            }
                        }
                    }
                }
            }
        }
        
        float[] pixelsToFloat(int[] pixels) {
            float[] output = new float[pixels.length];
            for (int i = 0; i < pixels.length; i++) {
                output[i] = pixels[i] / 255f;
            }
            return output;
        }
        
        List<Float> testingErrorList = new ArrayList<>();
        
        void test() {
            float sum = 0;
            for (int i = 0; i < 1000; i++) {
                int[] pixels = testImageList.get(i);
                int label = testLabelList.get(i);
                float[] x = pixelsToFloat(pixels);
                
                float[] y = new float[10];
                y[label] = 1;
                
                Tensor output = neuralNetwork.predictThreadSafe(x);
                sum += lossFunction.loss(Tensor.of(y), output);
            }
            float avg = sum / 1000f;
            testingErrorList.add(avg);
            if (testingErrorList.size() > 50)
                testingErrorList.remove(0);
        }
        
        Font font = new Font("Arial", Font.BOLD, 12);
        int paintIndex;
        
        int[] displayImagePixels;
        
        float highestValueOfArray(float[] array) {
            float max = 0;
            for (float a : array) {
                if (a > max)
                    max = a;
            }
            return max;
        }
        
        public void paint(Graphics graphics) {
            super.paint(graphics);
            
            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            int[] pixelsOfImage = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
            for (int i = 0; i < pixelsOfImage.length; i++) {
                int brightness = displayImagePixels[i];
                pixelsOfImage[i] = new Color(brightness, brightness, brightness).getRGB();
            }
            graphics.drawImage(image, 0, 0, getWidth() / 2, getHeight(), null);
            
            float[] x = pixelsToFloat(displayImagePixels);
            Tensor predict = neuralNetwork.predictThreadSafe(x);
            float[] y = predict.toFlatArray();
            
            float highestValue = highestValueOfArray(y);
            graphics.setColor(Color.white);
            graphics.setFont(font);
            int label = testLabelList.get(paintIndex);
            for (int i = 0; i < y.length; i++) {
                if (y[i] == highestValue) {
                    if (i == label)
                        graphics.setColor(Color.green);
                    else
                        graphics.setColor(Color.red);
                } else {
                    graphics.setColor(Color.white);
                }
                graphics.drawString(i + "=" + (int) (y[i] * 100d) + "%", (int) (10 + i / 10d * getWidth() / 2), getHeight() - 20);
            }
            
            drawLine(
                graphics,
                testingErrorList,
                getWidth() / 2,
                0,
                getWidth() / 2,
                getHeight()
            );
            
            if (!testingErrorList.isEmpty()) {
                graphics.setColor(Color.black);
                graphics.setFont(font);
                graphics.drawString(
                    "Error: " + (testingErrorList.get(testingErrorList.size() - 1)),
                    getWidth() / 2 + 10,
                    getHeight() - 20
                );
            }
        }
        
        void drawLine(Graphics graphics, List<Float> list, int x, int y, int width, int height) {
            int lastX2 = -1;
            int lastY2 = -1;
            graphics.setColor(Color.red);
            Graphics2D graphics2D = (Graphics2D) graphics;
            graphics2D.setStroke(new BasicStroke(2f));
            
            for (int i = 0; i < list.size(); i++) {
                float element = list.get(i);
                
                int x2 = (int) (x + (float) width / list.size() * i);
                int y2 = (int) (getHeight() - (y + (float) height * element));
                
                if (lastX2 != -1) {
                    graphics.drawLine(lastX2, lastY2, x2, y2);
                }
                
                lastX2 = x2;
                lastY2 = y2;
            }
        }
    }
}