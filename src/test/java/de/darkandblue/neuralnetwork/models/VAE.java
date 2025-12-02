package de.darkandblue.neuralnetwork.models;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.lossfunction.AbsoluteLoss;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.List;

public class VAE extends JFrame {
  static List<int[]> imageList;
  static List<Integer> labelList;
  
  static NeuralNetwork encoder = new NetworkBuilder()
      .distribution.xavier()
      .layer.dense(784, 200)
      .activation.leakyReLu()
      .layer.dense(200, 100)
      .activation.leakyReLu()
      .layer.dense(100, 80)
      .activation.leakyReLu()
      .layer.dense(80, 2)
//            .activation.tanh()
      .build();
  static NeuralNetwork decoder = new NetworkBuilder()
      .distribution.xavier()
      .layer.dense(2, 80)
      .activation.leakyReLu()
      .layer.dense(80, 100)
      .activation.leakyReLu()
      .layer.dense(100, 200)
      .activation.leakyReLu()
      .layer.dense(200, 784)
      .activation.sigmoid()
      .build();
  
  public static void main(String[] args) {
    imageList = MnistLoader.readImages();
    labelList = MnistLoader.readLabels();
    
    new VAE();
    new TwoDimVisualizer();
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
  
  static float transformRange(float value, float oldMin, float oldMax, float newMin, float newMax) {
    // Normalize value to [0, 1]
    float normalized = (value - oldMin) / (oldMax - oldMin);
    
    // Scale to new range
    return newMin + normalized * (newMax - newMin);
  }
  
  static void setPixels(int[] pixels, BufferedImage image) {
    int[] imagePixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
    
    for (int i = 0; i < pixels.length; i++) {
      int pixel = Math.min(Math.max(pixels[i], 0), 255);
      imagePixels[i] = new Color(pixel, pixel, pixel).getRGB();
    }
  }
  
  float learningRate = 0.00025f;
  
  public VAE() {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setLayout(null);
    
    setLayout(new BorderLayout());
    Scene scene = new Scene();
    scene.setMaximumSize(new Dimension(2000, 2000));
    add(scene, BorderLayout.CENTER);
    
    JSlider sliderLearningRate = new JSlider(0, 5000, (int) (5000f * learningRate * 3f));
    sliderLearningRate.setMaximumSize(new Dimension(1500, 40));
    sliderLearningRate.addChangeListener(e -> {
      learningRate = sliderLearningRate.getValue() / (float) sliderLearningRate.getMaximum() / 30f;
      updateTitle();
    });
    add(sliderLearningRate, BorderLayout.SOUTH);
    
    setSize(300, 200);
    setVisible(true);
    setLocationRelativeTo(null);
    
    updateTitle();
  }
  
  void updateTitle() {
    setTitle("learningRate: " + learningRate);
  }
  
  class Scene extends JPanel {
    private final static LossFunction decoderLossFunction = new AbsoluteLoss();
//        private final static LossFunction encoderLossFunction = new AbsoluteLoss();
    
    // TODO: wouldn't it be usefull if the loss function of an image gets determined by how much a number looks like a number? 
    // a algorithm would be usefull which compares the generated image and how it deviates from pixels near by from the original
    
    public Scene() {
      
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
      
      new Thread(() -> {
        while (true) {
          test();
        }
      }).start();
    }
    
    float testError;
    
    void test() {
      float sumTotal = 0;
      int iterations = 3000;
      for (int i = 0; i < iterations; i++) {
        int[] pixels = imageList.get(i);
        float[] x = pixelsToFloat(pixels);
        
        NumpyArray encoderOutput = encoder.predictThreadSafe(NumpyArray.of(x));
        NumpyArray decoderOutput = decoder.predictThreadSafe(encoderOutput);
        
        float[] totalError = decoderOutput.transpose().data[0];
        float sum = 0;
        for (int e = 0; e < totalError.length; e++) {
          sum += Math.abs(x[e] - totalError[e]);
        }
        sumTotal += sum / totalError.length;
      }
      testError = sumTotal / iterations;
    }
    
    int trainIndex;
    
    void train() {
      trainIndex++;
      trainIndex %= 60000;
      
      int[] pixels = imageList.get(trainIndex);
      float[] x = pixelsToFloat(pixels);
      
      NumpyArray encoderOutput = encoder.predict(NumpyArray.of(x));
      NumpyArray decoderOutput = decoder.predict(encoderOutput);
      
      NumpyArray grad = decoder.backpropagaton(decoderLossFunction, decoderOutput, x, learningRate);
      
      encoder.continueBackpropagation(grad, learningRate);
//            // making gradient ascent on the encoder
//            grad = grad.multiply(-1);
//
//            neuralNetworkEncoder.trainWithoutPredict(encoderLossFunction, encoderOutput, grad, learningRate * 0.5f);
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
      imageSize = Math.min(imageSize, getHeight());
      
      int[] pixels = imageList.get(thinkIndex);
      BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
      setPixels(pixels, image);
      graphics.drawImage(image, 0, 0, imageSize, imageSize, null);
      
      float[] inputs = pixelsToFloat(pixels);
      NumpyArray x = encoder.predictThreadSafe(inputs);
      
      int[] lowResPixels = floatToPixels(x.transpose().data[0]);
      image = new BufferedImage(7, 7, BufferedImage.TYPE_INT_RGB);
      setPixels(lowResPixels, image);
      graphics.drawImage(image, imageSize, 0, imageSize, imageSize, null);
      
      float[] y = decoder.predictThreadSafe(x).transpose().data[0];
      int[] predictedPixels = floatToPixels(y);
      image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
      setPixels(predictedPixels, image);
      graphics.drawImage(image, imageSize * 2, 0, imageSize, imageSize, null);
      
      graphics.setColor(Color.white);
      graphics.drawString("error: " + testError, 10, imageSize - 20);
    }
  }
  
  static class TwoDimVisualizer extends JPanel {
    TwoDimVisualizer() {
      JFrame frame = new JFrame();
      frame.add(this);
      frame.setSize(600, 600);
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
      frame.setVisible(true);
      frame.setLocationRelativeTo(null);
      setBackground(Color.darkGray);
      
      new Thread(() -> {
        while (true) {
          try {
            Thread.sleep(32);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
          repaint();
        }
      }).start();
    }
    
    float lastMinX = 9999, lastMaxX = -9999;
    float lastMinY = 9999, lastMaxY = -9999;
    
    @Override
    public void paint(Graphics graphics) {
      super.paint(graphics);
      
      float minX = 9999, maxX = -9999;
      float minY = 9999, maxY = -9999;
      for (int i = 0; i < 200; i++) {
        int[] pixels = imageList.get(i);
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        setPixels(pixels, image);
        
        float[] inputs = pixelsToFloat(pixels);
        NumpyArray output = encoder.predictThreadSafe(inputs);
        
        float x = output.data[0][0];
        float y = output.data[1][0];
        
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        
        x = transformRange(x, lastMinX, lastMaxX, 0, getWidth());
        y = transformRange(y, lastMinY, lastMaxY, 0, getHeight());
        
        graphics.drawImage(image, (int) x + 10, (int) y + 10, 20, 20, null);
      }
      this.lastMinX = minX;
      this.lastMaxX = maxX;
      this.lastMinY = minY;
      this.lastMaxY = maxY;
    }
  }
}