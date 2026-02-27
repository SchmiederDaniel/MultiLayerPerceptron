package models;

import models.dataset.Batch;
import models.dataset.DataSet;
import models.dataset.MNISTLoader;
import models.dataset.MnistAutoEncoderDataset;
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
    
    float learningRate = 0.0005f;
    
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
        NeuralNetwork autoencoder = new NetworkBuilder()
            .optimizer.adam()
            .distribution.xavier()
            // Encoder
            .layer.reshape(1, 28, 28)
            .layer.conv2D(1, 25, 7, 2, 0)
            .activation.gelu()
            .layer.conv2D(25, 25, 3, 2, 0)
            .activation.gelu()
            .layer.conv2D(25, 25, 3, 2, 0)
            .activation.gelu()
            .layer.flatten()
            // Bottleneck (use a moderate size to keep information)
            .layer.dense(100, 1)
            .activation.gelu()
            // Decoder
            .layer.dense(1, 300)
            .activation.gelu()
            .layer.dense(300, 400)
            .activation.gelu()
            .layer.dense(400, 400)
            .activation.gelu()
            .layer.dense(400, 784)
            .activation.gelu()
            .build();
        
        // TODO: wouldn't it be usefull if the loss function of an image gets determined by how much a number looks like a number? 
        // a algorithm would be usefull which compares the generated image and how it deviates from pixels near by from the original
        
        public Scene() {
            imageList = MNISTLoader.trainData();
            System.out.println(imageList.size());
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
                // Batch training using the new DataSet API
                int batchSize = 16;
                DataSet trainSet = new MnistAutoEncoderDataset(batchSize, true);
                while (true) {
                    for (Batch batch : trainSet) {
                        autoencoder.train(batch.inputs, batch.targets, learningRate);
                    }
                }
            }).start();
            
            new Thread(() -> {
                DataSet testSet = new MnistAutoEncoderDataset(128, false);
                while (true) {
                    test(testSet);
                }
            }).start();
        }
        
        float testError;
        
        void test(DataSet testSet) {
            float sumTotal = 0f;
            int count = 0;
            for (Batch batch : testSet) {
                Tensor[] inputs = batch.inputs;
                Tensor[] targets = batch.targets;
                for (int i = 0; i < inputs.length; i++) {
                    Tensor out = autoencoder.predictThreadSafe(inputs[i]);
                    float[] y = out.transpose().toFlatArray();
                    float[] x = targets[i].transpose().toFlatArray();
                    float sum = 0f;
                    for (int e = 0; e < y.length; e++) sum += Math.abs(x[e] - y[e]);
                    sumTotal += sum / y.length;
                    count++;
                    if (count >= 3000) break; // cap for speed similar to previous logic
                }
                if (count >= 3000) break;
            }
            if (count > 0) testError = sumTotal / count;
        }
        
        int thinkIndex;
        
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
            float[] y = autoencoder.predictThreadSafe(Tensor.of(inputs)).transpose().toFlatArray();
            int[] predictedPixels = floatToPixels(y);
            image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            setPixels(predictedPixels, image);
            // Use right two-thirds to show only reconstruction; middle panel kept blank for simplicity
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