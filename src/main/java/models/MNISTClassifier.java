package models;

import models.dataset.Batch;
import models.dataset.DataSet;
import models.dataset.MNISTLoader;
import models.dataset.MnistDataset;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.layer.Dense;
import neuralnetwork.layer.Layer;
import neuralnetwork.lossfunction.*;
import neuralnetwork.math.Matrix;
import neuralnetwork.math.Tensor;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
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
        setSize(1500, 500);
        add(new Scene());
        
        setVisible(true);
        setLocationRelativeTo(null);
    }
    
    class Scene extends JPanel {
        LossFunction lossFunction = new CategoricalCrossEntropy();
        NeuralNetwork neuralNetwork = new NetworkBuilder()
            .optimizer.adam()
            .distribution.xavier()
            .layer.reshape(1, 28, 28)
            .layer.conv2D(1, 20, 7, 2, 0)
            .activation.gelu()
            .layer.dropOut(0.25f)
            .layer.conv2D(20, 20, 3, 2, 0)
            .activation.gelu()
            .layer.dropOut(0.2f)
            .layer.conv2D(20, 20, 3, 2, 0)
            .activation.gelu()
            .layer.dropOut(0.2f)
            .layer.conv2D(20, 20, 3, 2, 0)
            .activation.gelu()
            .layer.dropOut(0.1f)
            .layer.flatten()
            .layer.dense(20, 40)
            .activation.gelu()
            .layer.dropOut(0.1f)
            .layer.dense(40, 10)
            .activation.softMax()
            .build();
        //        List<int[]> imageList = MNISTLoader.trainData();
//        List<Integer> labelList = MNISTLoader.trainLabels();
        List<int[]> testImageList = MNISTLoader.testData();
        List<Integer> testLabelList = MNISTLoader.testLabels();
        float learningRate = 0.0005f;
        int lastX = -1;
        int lastY = -1;
        Scene scene = this;
        
        public Scene() {
            setLayout(null);
            Button resetButton = new Button();
            resetButton.setLocation(10, 10);
            resetButton.setSize(25, 25);
            resetButton.addActionListener(e -> {
                Graphics2D graphics = ownImage.createGraphics();
                graphics.clearRect(0, 0, 28, 28);
                graphics.dispose();
            });
            add(resetButton);
            
            displayImagePixels = testImageList.get(paintIndex);
            setBackground(new Color(0, 0, 0, 0));
            
            addMouseMotionListener(new MouseMotionAdapter() {
                @Override
                public void mouseDragged(MouseEvent e) {
                    if (lastX != -1) {
                        Graphics2D graphics = ownImage.createGraphics();
                        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                        graphics.setColor(Color.white);
                        graphics.setStroke(new BasicStroke(1.5f));
                        graphics.drawLine(
                            (int) ((float) lastX / scene.getHeight() * 28f),
                            (int) ((float) lastY / scene.getHeight() * 28f),
                            (int) ((float) e.getX() / scene.getHeight() * 28f),
                            (int) ((float) e.getY() / scene.getHeight() * 28f)
                        );
                        repaint();
                    }
                    lastX = e.getX();
                    lastY = e.getY();
                }
            });
            addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    super.mouseClicked(e);
                    lastX = e.getX();
                    lastY = e.getY();
                    repaint();
                }
                
                @Override
                public void mouseReleased(MouseEvent e) {
                    super.mouseReleased(e);
                    lastX = -1;
                    lastY = -1;
                    repaint();
                }
            });
            
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
        
        //        int trainIndex;
        DataSet trainSet = new MnistDataset(10, true, 0);
        
        void train() {
            for (Batch batch : trainSet) {
                neuralNetwork.train(batch.inputs, batch.targets, learningRate);
            }

//            trainIndex++;
//            trainIndex %= imageList.size();
//            
//            int[] pixels = imageList.get(trainIndex);
//            int label = labelList.get(trainIndex);
//            
//            float[] x = pixelsToFloat(pixels);
//            float[] y = new float[10];
//            y[label] = 1;
//            
//            neuralNetwork.trainSingle(lossFunction, Tensor.of(x), Tensor.of(y), learningRate);
//            
//            for (Layer layer : neuralNetwork.layers) {
//                if (layer instanceof Dense dense) {
//                    if (trainIndex % 1000 == 0) { // remove faded out graidents
//                        Matrix w = dense.W;
//                        float[][] values = w.getValues();
//                        for (int rowIndex = 0; rowIndex < values.length; rowIndex++) {
//                            float[] row = values[rowIndex];
//                            for (int colIndex = 0; colIndex < values[0].length; colIndex++) {
//                                float v = row[colIndex];
//                                values[rowIndex][colIndex] = v;
//                            }
//                        }
//                    }
//                }
//            }
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
        
        BufferedImage ownImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        int[] ownImagePixels = ((DataBufferInt) ownImage.getRaster().getDataBuffer()).getData();
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
            Graphics2D graphics2D = (Graphics2D) graphics;
            graphics2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            graphics2D.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            graphics.setColor(Color.white);
            graphics.fillRect(0, 0, getWidth(), getHeight());
            
            drawMNISTImage(graphics);
            drawOwnImage(graphics);
            
            drawLine(
                graphics,
                testingErrorList,
                getHeight() * 2,
                0,
                getWidth(),
                getHeight()
            );
            
            if (!testingErrorList.isEmpty()) {
                graphics.setColor(Color.black);
                graphics.setFont(font);
                graphics.drawString(
                    "Loss: " + (testingErrorList.getLast()),
                    getHeight() * 2 + 10,
                    getHeight() - 20
                );
            }
            
            super.paint(graphics);
        }
        
        void drawOwnImage(Graphics graphics) {
            graphics.drawImage(ownImage, 0, 0, getHeight(), getHeight(), null);
            
            float[] x = new float[28 * 28];
            for (int i = 0; i < x.length; i++) {
                x[i] = new Color(ownImagePixels[i]).getRed() / 255f;
            }
            Tensor predict = neuralNetwork.predictThreadSafe(Tensor.of(x));
            float[] y = predict.toFlatArray();
            
            int label = -1;
            drawClassificationText(graphics, 0, y, label);
        }
        
        void drawMNISTImage(Graphics graphics) {
            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            int[] pixelsOfImage = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
            for (int i = 0; i < pixelsOfImage.length; i++) {
                int brightness = displayImagePixels[i];
                pixelsOfImage[i] = new Color(brightness, brightness, brightness).getRGB();
            }
            graphics.drawImage(image, getHeight(), 0, getHeight(), getHeight(), null);
            
            float[] x = pixelsToFloat(displayImagePixels);
            Tensor predict = neuralNetwork.predictThreadSafe(x);
            float[] y = predict.toFlatArray();
            
            int label = testLabelList.get(paintIndex);
            drawClassificationText(graphics, getHeight(), y, label);
        }
        
        void drawClassificationText(Graphics graphics, int posX, float[] y, int label) {
            graphics.setColor(Color.white);
            graphics.setFont(font);
            float highestValue = highestValueOfArray(y);
            for (int i = 0; i < y.length; i++) {
                if (y[i] == highestValue) {
                    if (i == label || label == -1)
                        graphics.setColor(Color.green);
                    else
                        graphics.setColor(Color.red);
                } else {
                    graphics.setColor(Color.white);
                }
                graphics.drawString(i + "=" + (int) (y[i] * 100d) + "%", (int) (10 + i / 11d * getHeight()) + posX, getHeight() - 20);
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