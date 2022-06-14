package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.lossfunction.BinaryCrossEntropy;
import de.darkandblue.neuralnetwork.lossfunction.OwnLoss;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.util.List;

public class UpscaleAutoEncoder extends JFrame {
  public static void main(String[] args) throws IOException {
    new UpscaleAutoEncoder();
  }
  
  public UpscaleAutoEncoder() throws IOException {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    
    add(new Scene());
    
    setSize(1200, 600);
    setVisible(true);
    setLocationRelativeTo(null);
    
  }
  
  class Scene extends JPanel {
    List<int[]> imageList;
    List<Integer> labelList;
    BinaryCrossEntropy lossFunction = new BinaryCrossEntropy();
    NeuralNetwork neuralNetwork = new NetworkBuilder()
      .dense(7 * 7, 392)
      .sigmoid()
      .dense(392, 784)
      .sigmoid()
      .build();
    
    public Scene() throws IOException {
      imageList = MnistLoader.readImages();
      labelList = MnistLoader.readLabels();
      
      startAsyncThreads();
    }
    
    public void startAsyncThreads() {
      new Thread(() -> {
        while (true) {
          try {
            Thread.sleep(1300);
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
    }
    
    int trainIndex;
    
    void train() {
      trainIndex++;
      
      int[] pixels = imageList.get(trainIndex);
      double[] x = pixelsToDouble(downScalePixels(pixels));
      double[] y = pixelsToDouble(pixels);
      
      neuralNetwork.trainSingle(lossFunction, x, y, 0.5, false);
    }
    
    static double[] pixelsToDouble(int[] pixels) {
      double[] output = new double[pixels.length];
      for (int i = 0; i < pixels.length; i++) {
        output[i] = pixels[i] / 255d;
      }
      return output;
    }
    
    static int[] doubleToPixels(double[] pixels) {
      int[] output = new int[pixels.length];
      for (int i = 0; i < pixels.length; i++) {
        output[i] = (int) (pixels[i] * 255d);
      }
      return output;
    }
    
    static int[] downScalePixels(int[] pixels) {
      int[] resized = new int[7 * 7];
      
      for (int i = 0; i < pixels.length; i++) {
        int x = i % 28;
        int y = i / 28;
        
        x /= 4;
        y /= 4;
        int i2 = x + y * 7;
        
        resized[i2] += pixels[i];
      }
      
      for (int i = 0; i < resized.length; i++) {
        resized[i] = (int) (resized[i] / (4d * 4d));
        resized[i] = Math.min(Math.max(resized[i], 0), 255);
      }
      
      return resized;
    }
    
    int thinkIndex;
    
    public void paint(Graphics graphics) {
      super.paint(graphics);
      thinkIndex++;
      thinkIndex %= 60000;
      
      int[] pixels = imageList.get(trainIndex);
      int[] lowResPixels = downScalePixels(pixels);
      BufferedImage image = new BufferedImage(7, 7, BufferedImage.TYPE_INT_RGB);
      setPixels(lowResPixels, image);
      graphics.drawImage(image, 0, 0, getWidth() / 2, getHeight(), null);
      
      double[] x = pixelsToDouble(lowResPixels);
      
      double[] y = neuralNetwork.predict(x).transpose().data[0];
      int[] predictedPixels = doubleToPixels(y);
      image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
      setPixels(predictedPixels, image);
      graphics.drawImage(image, getWidth() / 2, 0, getWidth() / 2, getHeight(), null);
    }
    
    static void setPixels(int[] pixels, BufferedImage image) {
      int[] imagePixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
      
      for (int i = 0; i < pixels.length; i++) {
        int pixel = Math.min(Math.max(pixels[i], 0), 255);
        imagePixels[i] = new Color(pixel, pixel, pixel).getRGB();
      }
    }
  }
}