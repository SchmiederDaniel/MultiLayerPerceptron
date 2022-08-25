package de.darkandblue.neuralnetwork.diffusion;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.lossfunction.BinaryCrossEntropy;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.naming.BinaryRefAddr;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class DiffusionTest extends JFrame {
  public static void main(String[] args) {
    new DiffusionTest();
  }
  
  public DiffusionTest() {
    
    Scene scene = new Scene();
    add(scene);
    
    addComponentListener(new ComponentAdapter() {
      public void componentResized(ComponentEvent e) {
        scene.setSize(getWidth(), getWidth());
      }
    });
    
    setSize(1600, 800);
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setVisible(true);
    setLocationRelativeTo(null);
    
  }
  
  class Scene extends JPanel {
    NeuralNetwork neuralNetwork = new NetworkBuilder()
      .dense(28 * 28, 28 * 28)
      .sigmoid()
      .dense(28 * 28, 28 * 28)
      .sigmoid()
      .build();
    private static final int steps = 1000;
    
    List<int[]> images = MnistLoader.readImages();
    List<Integer> labels = MnistLoader.readLabels();
    LossFunction lossFunction = new BinaryCrossEntropy();
    double learningRate = 0.001;
    
    public Scene() {
      startAsyncLoops();
    }
    
    void startAsyncLoops() {
      new Thread(() -> {
        while (true) {
          train();
        }
      }).start();
      new Thread(() -> {
        while (true) {
          predict();
          try {
            Thread.sleep(3000);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        }
      }).start();
      new Thread(() -> {
        while (true) {
          repaint();
          try {
            Thread.sleep(16);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        }
      }).start();
    }
    
    void train() {
      int trainIndex = ThreadLocalRandom.current().nextInt(images.size());
      
      int[] image = images.get(trainIndex);
      int label = labels.get(trainIndex);
  
      int[] noise = new int[28 * 28];
      for (int i = 0; i < noise.length; i++) {
        noise[i] = (int) (Math.random() * 255);
      }
  
      int step = ThreadLocalRandom.current().nextInt(steps);
  
      double[] noiseFrom = new double[28 * 28];
      double[] noiseTo = new double[28 * 28];
      for (int i = 0; i < image.length; i++) {
        int diff = noise[i] - image[i];
        double noiseAtStep = diff / (step / steps);
        noiseFrom[i] = (image[i] + noiseAtStep - diff / steps) / 255d;
        noiseTo[i] = (image[i] + noiseAtStep) / 255d;
      }
      
      neuralNetwork.trainSingle(lossFunction, noiseFrom, noiseTo, learningRate, false);
    }
  
    BufferedImage noisyImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
    int[] noisyPixels = ((DataBufferInt) noisyImage.getRaster().getDataBuffer()).getData();
    BufferedImage deNoisedImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
    int[] denoisedPixels = ((DataBufferInt) deNoisedImage.getRaster().getDataBuffer()).getData();
    int predictIndex;
    
    void predict() {
      predictIndex++;
      predictIndex %= images.size();
      
      int[] image = images.get(predictIndex);
  
      int[] noise = new int[28 * 28];
      for (int i = 0; i < noise.length; i++) {
        noise[i] = (int) (Math.random() * 255d);
      }
  
      for (int i = 0; i < image.length; i++) {
        int diff = noise[i] - image[i];
        int brightness = (int) (image[i] + diff / 2d);
        noisyPixels[i] = new Color(brightness, brightness, brightness).getRGB();
      }
  
      double[] cache = intArrayToDouble(noise);
      for (int i = steps / 2; i < steps; i++) {
        cache = neuralNetwork.predictThreadSafe(cache).transpose().data[0];
      }
      for (int i = 0; i < image.length; i++) {
        int brightness = (int) (cache[i] * 255d);
        denoisedPixels[i] = new Color(brightness, brightness, brightness).getRGB();
      }
    }
    
    @Override
    public void paint(Graphics graphics) {
      super.paint(graphics);
      
      graphics.drawImage(noisyImage, 0, 0, getWidth() / 2, getHeight(), null);
      
      graphics.drawImage(deNoisedImage, getWidth() / 2, 0, getWidth() / 2, getHeight(), null);
    }
  }
  
  private static double[] intArrayToDouble(int[] array) {
    double[] arrayDouble = new double[array.length];
    for (int i = 0; i < array.length; i++) {
      arrayDouble[i] = array[i] / 255d;
    }
    return arrayDouble;
  }
  
  private static int[] doubleArrayToInt(double[] array) {
    int[] arrayInt = new int[array.length];
    for (int i = 0; i < array.length; i++) {
      arrayInt[i] = (int) (array[i] * 255d);
    }
    return arrayInt;
  }
  
  private static double map(double value, double minFrom, double maxFrom, double minTo, double maxTo) {
    return (value - minFrom) / (maxFrom - minFrom) * (maxTo - minTo) + minTo;
  }
  
}