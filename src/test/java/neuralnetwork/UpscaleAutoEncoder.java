package neuralnetwork;

import neuralnetwork.lossfunction.BinaryCrossEntropy;
import neuralnetwork.lossfunction.LossFunction;
import neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.util.List;

public class UpscaleAutoEncoder extends JFrame {
  public static void main(String[] args) throws IOException {
    new UpscaleAutoEncoder();
  }
  float learningRate = 0;
  
  public UpscaleAutoEncoder() throws IOException {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setLayout(null);
  
    Scene scene = new Scene();
    add(scene);
    
    JSlider sliderLearningRate = new JSlider(0, 5000, 0);
    sliderLearningRate.addChangeListener(e -> {
      learningRate = sliderLearningRate.getValue() / (float) sliderLearningRate.getMaximum() * 5;
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
      .layer.dense(7 * 7, 784)
      .activation.sigmoid()
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
      float[] x = pixelsTofloat(downScalePixels(pixels));
      float[] y = pixelsTofloat(pixels);
      
      neuralNetwork.trainSingle(lossFunction, x, y, learningRate, false);
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
    
    int[] downScalePixels(int[] pixels) {
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
      
      int[] lowResPixels = downScalePixels(pixels);
      image = new BufferedImage(7, 7, BufferedImage.TYPE_INT_RGB);
      setPixels(lowResPixels, image);
      graphics.drawImage(image, imageSize, 0, imageSize, imageSize, null);
      
      float[] x = pixelsTofloat(lowResPixels);
      
      float[] y = neuralNetwork.predictThreadSafe(x).transpose().data[0];
      int[] predictedPixels = floatToPixels(y);
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