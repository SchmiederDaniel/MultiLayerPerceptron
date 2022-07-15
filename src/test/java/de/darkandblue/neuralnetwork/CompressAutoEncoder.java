package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.lossfunction.BinaryCrossEntropy;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.util.List;

public class CompressAutoEncoder extends JFrame {
  public static void main(String[] args) throws IOException {
    new CompressAutoEncoder();
  }
  
  double learningRate = 0;
  
  public CompressAutoEncoder() throws IOException {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setLayout(null);
    
    Scene scene = new Scene();
    add(scene);
    
    JSlider sliderLearningRate = new JSlider(0, 5000, 0);
    sliderLearningRate.addChangeListener(e -> {
      learningRate = sliderLearningRate.getValue() / (double) sliderLearningRate.getMaximum();
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
    setTitle("learningRate: " + learningRate);
  }
  
  class Scene extends JPanel {
    List<int[]> imageList;
    List<Integer> labelList;
    LossFunction lossFunction = new BinaryCrossEntropy();
    NeuralNetwork neuralNetwork = new NetworkBuilder()
      .dense(784, 30)
      .sigmoid()
      .dense(30, 7)
      .sigmoid()
      .dense(7, 30)
      .sigmoid()
      .dense(30, 784)
      .sigmoid()
      .build();
    
    // TODO: wouldn't it be usefull if the loss function of an image gets determined by how much a number looks like a number? 
    // a algorithm would be usefull which compares the generated image and how it deviates from pixels near by from the original
    
    public Scene() throws IOException {
      imageList = MnistLoader.readImages();
      labelList = MnistLoader.readLabels();
      
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
    }
    
    int trainIndex;
    
    void train() {
      trainIndex++;
      trainIndex %= 60000;
      
      int[] pixels = imageList.get(trainIndex);
      double[] x = pixelsToDouble(pixels);
      
      neuralNetwork.trainSingle(lossFunction, x, x, learningRate, false);
    }
    
    double[] pixelsToDouble(int[] pixels) {
      double[] output = new double[pixels.length];
      for (int i = 0; i < pixels.length; i++) {
        output[i] = pixels[i] / 255d;
      }
      return output;
    }
    
    int[] doubleToPixels(double[] pixels) {
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
      
      double[] x = pixelsToDouble(pixels);
      
      NumpyArray output = NumpyArray.of(x);
      for (int i = 0; i < neuralNetwork.layerArray.length / 2; i++) {
        Layer layer = neuralNetwork.layerArray[i].deepCopy();
        output = layer.forward(output);
      }
      
      int[] lowResPixels = doubleToPixels(output.transpose().data[0]);
      image = new BufferedImage(7, 7, BufferedImage.TYPE_INT_RGB);
      setPixels(lowResPixels, image);
      graphics.drawImage(image, imageSize, 0, imageSize, imageSize, null);
      
      double[] y = neuralNetwork.predictThreadSafe(x).transpose().data[0];
      int[] predictedPixels = doubleToPixels(y);
      image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
      setPixels(predictedPixels, image);
      graphics.drawImage(image, imageSize * 2, 0, imageSize, imageSize, null);
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