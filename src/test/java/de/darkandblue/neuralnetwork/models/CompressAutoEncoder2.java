package de.darkandblue.neuralnetwork.models;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.lossfunction.AbsoluteLoss;
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
  
  float learningRate = 0.025f;
  
  public CompressAutoEncoder2() {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setLayout(null);
    
    Scene scene = new Scene();
    add(scene);
    
    JSlider sliderLearningRate = new JSlider(0, 5000, (int) (5000d * learningRate * 3d));
    sliderLearningRate.addChangeListener(e -> {
      learningRate = sliderLearningRate.getValue() / (float) sliderLearningRate.getMaximum() / 3f;
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
    private final static LossFunction decoderLossFunction = new AbsoluteLoss();
    private final static LossFunction encoderLossFunction = new AbsoluteLoss();
    
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
    
    float testError;
    void test() {
      float sumTotal = 0;
      int iterations = 3000;
      for (int i = 0; i < iterations; i++) {
        int[] pixels = imageList.get(i);
        float[] x = pixelsTofloat(pixels);
  
        NumpyArray encoderOutput = neuralNetworkEncoder.predictThreadSafe(NumpyArray.of(x));
        NumpyArray decoderOutput = neuralNetworkDecoder.predictThreadSafe(encoderOutput);
  
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
      float[] x = pixelsTofloat(pixels);
      
      NumpyArray encoderOutput = neuralNetworkEncoder.predict(NumpyArray.of(x));
      NumpyArray decoderOutput = neuralNetworkDecoder.predict(encoderOutput);
  
      NumpyArray grad = neuralNetworkDecoder.backwardWithoutTrain(decoderLossFunction, decoderOutput, NumpyArray.of(x), learningRate);
  
      // making gradient ascent on the encoder
      grad = grad.multiply(-1);
  
      neuralNetworkEncoder.backwardWithoutTrain(encoderLossFunction, encoderOutput, grad, learningRate * 0.5f);
    }
    
    float[] pixelsTofloat(int[] pixels) {
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
      
      float[] inputs = pixelsTofloat(pixels);
      NumpyArray x = neuralNetworkEncoder.predictThreadSafe(inputs);
      
      float[] y = neuralNetworkDecoder.predictThreadSafe(x).transpose().data[0];
      int[] predictedPixels = floatToPixels(y);
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