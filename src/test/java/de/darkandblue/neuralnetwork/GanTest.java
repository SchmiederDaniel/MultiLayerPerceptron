package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.lossfunction.BasicLoss;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.Random;

public class GanTest extends JFrame {
  public static void main(String[] args) throws IOException {
    new GanTest();
  }
  
  public GanTest() throws IOException {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    
    Scene scene = new Scene();
    add(scene);
    
    setSize(800, 800);
    setVisible(true);
    setLocationRelativeTo(null);
  }
  
  class Scene extends JPanel {
    List<int[]> imageList;
    List<Integer> labelList;
    int noiseCount = 64;
    
    GANetwork network = new GANBuilder()
      .denseG(noiseCount, 200)
      .sigmoidG()
      .denseG(200, 784)
      .sigmoidG()
      .denseD(784, 200)
      .sigmoidD()
      .denseD(200, 1)
      .sigmoidD()
      .build();
    // TODO print the loss functions of generator and discriminator to check why it isnt working
    
    public Scene() throws IOException {
      imageList = MnistLoader.readImages();
      labelList = MnistLoader.readLabels();
  
      startThreads();
    }
    
    void startThreads() {
      new Thread(()->{
        while(true) {
          train();
        }
      }).start();
  
      new Thread(()->{
        while(true) {
          try {
            Thread.sleep(16);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
          repaint();
        }
      }).start();
    }
    
    int trainIndex = 0;
    void train() {
      trainIndex++;
      if (trainIndex >= imageList.size())
        trainIndex = 0;
      
      int[] pixels = imageList.get(trainIndex);
      double[] realArray = pixelsToDouble(pixels);
  
      double[] noise = generateNoise(noiseCount);
      
      network.trainSingle(realArray, noise, 0.001d);
    }
    
    int predictIndex;
    public void paint(Graphics graphics) {
      super.paint(graphics);
      predictIndex++;
      if(predictIndex > imageList.size())
        predictIndex = 0;
      
//      double[] noise = generateNoise(noiseCount);
      double[] noise = new double[noiseCount];
      Random random = new Random();
      for (int i = 0; i < noiseCount; i++) {
        noise[i] = random.nextDouble(0, 1);
      }
  
      double[] output = network.predictGeneratorThreadSafe(noise).transpose().data[0];
      int[] pixels = doubleToPixels(output);
  
      BufferedImage bufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
      for (int x = 0; x < 28; x++) {
        for (int y = 0; y < 28; y++) {
          int brightness = pixels[x + y * 28];
          bufferedImage.setRGB(x, y, new Color(brightness, brightness, brightness).getRGB());
        }
      }
      graphics.drawImage(bufferedImage, 0, 0, getWidth(), getHeight(), null);
    }
  
    static double[] pixelsToDouble(int[] pixels) {
      double[] result = new double[pixels.length];
      for (int i = 0; i < pixels.length; i++) {
//        result[i] = map(pixels[i], 0, 255d, -1d, 1d);
        result[i] = map(pixels[i], 0, 255d, 0d, 1d);
      }
      return result;
    }
  
    public static double map(double value, double minFrom, double maxFrom, double minTo, double maxTo) {
      return (value - minFrom) / (maxFrom - minFrom) * (maxTo - minTo) + minTo;
    }
  
    static int[] doubleToPixels(double[] pixels) {
      int[] result = new int[pixels.length];
      for (int i = 0; i < pixels.length; i++) {
//        result[i] = (int) map(pixels[i], -1, 1, 0, 255);
        result[i] = (int) map(pixels[i], 0, 1, 0, 255);
      }
      return result;
    }
    
    static double[] generateNoise(int length) {
      double[] noise = new double[length];
      for (int i = 0; i < length; i++)
        noise[i] = Math.random();
      return noise;
    }
  }
}