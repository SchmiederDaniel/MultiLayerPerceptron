package de.darkandblue.neuralnetwork.networks;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.lossfunction.BinaryCrossEntropy;
import de.darkandblue.neuralnetwork.lossfunction.LinearLoss;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.List;

public class CompressAutoEncoder2 extends JFrame {
  public static void main(String[] args) {
    new CompressAutoEncoder2();
  }
  
  double learningRate = 0.025;
  
  public CompressAutoEncoder2() {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setLayout(null);
    
    Scene scene = new Scene();
    add(scene);
    
    JSlider sliderLearningRate = new JSlider(0, 5000, (int) (5000d * learningRate * 3d));
    sliderLearningRate.addChangeListener(e -> {
      learningRate = sliderLearningRate.getValue() / (double) sliderLearningRate.getMaximum() / 3d;
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
    NeuralNetwork neuralNetworkEncoder = new NetworkBuilder()
      .layer.dense(784, 80)
      .activation.sigmoid()
      .layer.dense(80, 80)
      .activation.sigmoid()
      .layer.dense(80, 80)
      .activation.sigmoid()
      .build();
    NeuralNetwork neuralNetworkDecoder = new NetworkBuilder()
      .layer.dense(80, 80)
      .activation.sigmoid()
      .layer.dense(80, 80)
      .activation.sigmoid()
      .layer.dense(80, 784)
      .activation.sigmoid()
      .build();
    private final static LossFunction decoderLossFunction = new LinearLoss();
    private final static LossFunction encoderLossFunction = new LinearLoss();
    
    // TODO: wouldn't it be usefull if the loss function of an image gets determined by how much a number looks like a number? 
    // a algorithm would be usefull which compares the generated image and how it deviates from pixels near by from the original
    
    public Scene() {
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
  
      new Thread(() -> {
        try {
          Thread.sleep(10);
        } catch (InterruptedException e) {
          throw new RuntimeException(e);
        }
        while (true) {
          test();
        }
      }).start();
    }
    
    double testError;
    void test() {
      double sumTotal = 0;
      int iterations = 3000;
      for (int i = 0; i < iterations; i++) {
        int[] pixels = imageList.get(i);
        double[] x = pixelsToDouble(pixels);
  
        NumpyArray encoderOutput = neuralNetworkEncoder.predictThreadSafe(NumpyArray.of(x));
        NumpyArray decoderOutput = neuralNetworkDecoder.predictThreadSafe(encoderOutput);
  
        double[] totalError = decoderOutput.transpose().data[0];
        double sum = 0;
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
      double[] x = pixelsToDouble(pixels);
      
      NumpyArray encoderOutput = neuralNetworkEncoder.predict(NumpyArray.of(x));
      NumpyArray decoderOutput = neuralNetworkDecoder.predict(encoderOutput);
  
      NumpyArray grad = neuralNetworkDecoder.trainWithoutPredict(decoderLossFunction, decoderOutput, x, learningRate);
  
      // making gradient ascent on the encoder
      grad = grad.multiplyScalar(-1);
  
      neuralNetworkEncoder.trainWithoutPredict(encoderLossFunction, encoderOutput, grad, learningRate * 0.5d);
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
      
      double[] inputs = pixelsToDouble(pixels);
      NumpyArray x = neuralNetworkEncoder.predictThreadSafe(inputs);
      
      double[] y = neuralNetworkDecoder.predictThreadSafe(x).transpose().data[0];
      int[] predictedPixels = doubleToPixels(y);
      image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
      setPixels(predictedPixels, image);
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