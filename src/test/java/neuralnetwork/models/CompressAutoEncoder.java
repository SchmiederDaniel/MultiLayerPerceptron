package neuralnetwork.models;

import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.lossfunction.AbsoluteLoss;
import neuralnetwork.lossfunction.LossFunction;
import neuralnetwork.lossfunction.MeanSquareError;
import neuralnetwork.math.NumpyArray;
import neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.List;

public class CompressAutoEncoder extends JFrame {
  public static void main(String[] args) {
    new CompressAutoEncoder();
  }
  
  float learningRate = 0.00001f;
  
  public CompressAutoEncoder() {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setLayout(null);
    
    Scene scene = new Scene();
    add(scene);
    
    JSlider sliderLearningRate = new JSlider(0, 5000, (int) (5000f * learningRate * 3f));
    sliderLearningRate.addChangeListener(e -> {
      learningRate = sliderLearningRate.getValue() / (float) sliderLearningRate.getMaximum() / 3000f;
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
    
    setSize(800, 400);
    setVisible(true);
    setLocationRelativeTo(null);
    
    updateTitle();
  }
  
  void updateTitle() {
    setTitle("learningRate: " + String.format("%.8f", learningRate));
  }
  
  class Scene extends JPanel {
    List<int[]> imageList;
    List<Integer> labelList;
    NeuralNetwork encoder = new NetworkBuilder()
//        .distribution.customDistribution(0.3f, 0.31f, -0.3f, 0.3f)
        .distribution.heNormal()
        .layer.conv2D(1, 28, 28, 5, 5, 2, 10)
        .activation.leakyReLu()
        .layer.conv2D(10, 12, 12, 3, 3, 1, 5)
        .activation.leakyReLu()
        .layer.conv2D(5, 10, 10, 3, 3, 1, 5)
        .activation.leakyReLu()
        .layer.conv2D(5, 8, 8, 3, 3, 1, 5)
//        .activation.leakyReLu()
//        .layer.conv2D(5, 6, 6, 3, 3, 1, 3)
        .activation.leakyReLu()
        .layer.flatten()
        .layer.transpose()
        .layer.dense(200)
        .activation.leakyReLu()
        .layer.dense(200, 1)
//        .distribution.xavier()
        .build();
    NeuralNetwork decoder = new NetworkBuilder()
//        .distribution.xavier()
//        .layer.fourier(1, 5, 2f)
//        .distribution.xavier()
//        .layer.dense(256)
        .distribution.heNormal()
//        .distribution.customDistribution(0, 0.01f, -0.1f, 0.1f)
//        .distribution.customDistribution(0.1f, 0.11f, -0.3f, 0.3f)
        .layer.dense(1, 256)
        .activation.leakyReLu()
        .layer.reshape(1, 16, 16)
        .layer.conv2D(1, 16, 16, 3, 3, 1, 10)
        .activation.leakyReLu()
        .layer.conv2D(10, 14, 14, 3, 3, 1, 10)
        .activation.leakyReLu()
        .layer.flatten()
        .layer.transpose()
        .distribution.xavier()
        .layer.dense(784)
        .activation.sigmoid()
        .build();
    private final static LossFunction decoderLossFunction = new AbsoluteLoss();
    
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
        float[] floatPixel = pixelsToFloat(pixels);
        
        NumpyArray x = NumpyArray.of(floatPixel);
        NumpyArray x2d = x.reshape(1, 28, 28);
        NumpyArray encoderOutput = encoder.predictThreadSafe(x2d);
        NumpyArray decoderOutput = decoder.predictThreadSafe(encoderOutput);
        
        float[] totalError = decoderOutput.transpose().data[0][0];
        float sum = 0;
        for (int e = 0; e < totalError.length; e++) {
          sum += Math.abs(floatPixel[e] - totalError[e]);
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
      NumpyArray x = NumpyArray.of(pixelsToFloat(pixels));
      NumpyArray x2d = x.reshape(1, 28, 28);
      
      NumpyArray encoderOutput = encoder.predict(x2d);
      NumpyArray decoderOutput = decoder.predict(encoderOutput);
      
      NumpyArray grad = decoder.backpropagaton(decoderLossFunction, decoderOutput, x, learningRate);
      
      encoder.continueBackpropagation(grad, learningRate);
    }
    
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
      
      float[] inputs = pixelsToFloat(pixels);
      NumpyArray x = NumpyArray.of(inputs);
      NumpyArray x2d = x.reshape(1, 28, 28);
      x = encoder.predictThreadSafe(x2d);
      
      int[] lowResPixels = floatToPixels(x.transpose().data[0][0]);
      image = new BufferedImage(7, 7, BufferedImage.TYPE_INT_RGB);
      setPixels(lowResPixels, image);
      graphics.drawImage(image, imageSize, 0, imageSize, imageSize, null);
      
      float[] y = decoder.predictThreadSafe(x).transpose().data[0][0];
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