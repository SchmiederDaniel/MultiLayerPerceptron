package de.darkandblue.neuralnetwork.models;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.lossfunction.AbsoluteLoss;
import de.darkandblue.neuralnetwork.lossfunction.BinaryCrossEntropy;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class AutoGan extends JFrame {
  public static void main(String[] args) {
    new AutoGan();
  }
  
  Scene scene;
  
  public AutoGan() {
    setTitle("AutoGan");
    
    scene = new Scene();
    add(scene);
    
    setSize(1000, 1000);
    setVisible(true);
    setLocationRelativeTo(null);
    
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    
    startAsyncThreads();
  }
  
  void startAsyncThreads() {
    new Thread(() -> {
      while (true) {
        scene.think();
        try {
          Thread.sleep(1500);
        } catch (InterruptedException e) {
          throw new RuntimeException(e);
        }
      }
    }).start();
  
    new Thread(() -> {
      while (true) {
        scene.train();
      }
    }).start();
  
    new Thread(() -> {
      while (true) {
        scene.test();
      }
    }).start();
    
    new Thread(() -> {
      while (true) {
        scene.repaint();
        try {
          Thread.sleep(16);
        } catch (InterruptedException e) {
          throw new RuntimeException(e);
        }
      }
    }).start();
  }
  
  class Scene extends JPanel {
    NeuralNetwork encoder;
    NeuralNetwork decoder;
    NeuralNetwork discriminator;
    int imageResolution = 28;
    int[][] images;
    int[] labels;
    LossFunction encoderLoss = new AbsoluteLoss();
    LossFunction decoderLoss = new AbsoluteLoss();
    LossFunction discriminatorLoss = new BinaryCrossEntropy();
    float learningRate = 0.1f;
    
    public Scene() {
      images = MnistLoader.readImages().stream().toArray(int[][]::new);
      if (imageResolution != 28)
        for (int i = 0; i < images.length; i++)
          images[i] = downScale(images[i], imageResolution);
      labels = MnistLoader.readLabels().stream().mapToInt(i -> i).toArray();
      
      encoder = new NetworkBuilder()
        .layer.dense(imageResolution * imageResolution, 60)
        .activation.sigmoid()
        .layer.dense(60, 60)
        .activation.sigmoid()
        .build();
      
      decoder = new NetworkBuilder()
        .layer.dense(60, 60)
        .activation.sigmoid()
        .layer.dense(60, imageResolution * imageResolution)
        .activation.sigmoid()
        .build();
      
      discriminator = new NetworkBuilder()
        .layer.dense(imageResolution * imageResolution + imageResolution * imageResolution, 50)
        .activation.sigmoid()
        .layer.dense(50, 1)
        .activation.sigmoid()
        .build();
    }
    List<Float> data1 = new ArrayList<>();
    List<Float> data2 = new ArrayList<>();
    void test() {
      float sum1 = 0;
      float sum2 = 0;
      for (int i = 0; i < 200; i++) {
        int[] pixels = images[trainIndex];
        float[] realData = pixelsToFloat(pixels);
  
        float[] encoded = encoder.predict(NumpyArray.of(realData)).transpose().data[0];
        float[] decoded = decoder.predict(NumpyArray.of(encoded)).transpose().data[0];
  
        boolean order = Math.random() > 0.5d;
        float[] combined = new float[realData.length * 2];
        System.arraycopy(order ? realData : decoded, 0, combined, 0, imageResolution * imageResolution);
        System.arraycopy(order ? decoded : realData, 0, combined, imageResolution * imageResolution, imageResolution * imageResolution);
        float discriminated = discriminator.predict(NumpyArray.of(combined)).data[0][0];
  
        sum1 += order ? discriminated : 1 - discriminated;
  
        order = !order;
        System.arraycopy(order ? realData : decoded, 0, combined, 0, imageResolution * imageResolution);
        System.arraycopy(order ? decoded : realData, 0, combined, imageResolution * imageResolution, imageResolution * imageResolution);
        discriminated = discriminator.predict(NumpyArray.of(combined)).data[0][0];
  
        sum2 += order ? 1 - discriminated : discriminated;
      }
  
      data1.add(sum1 / 200f);
      data2.add(sum2 / 200f);
      
      System.out.println("" + sum1 / 200d + " " + sum2 / 200d);
  
      if(data1.size() > 400)
        data1.remove(0);
      if(data2.size() > 400)
        data2.remove(0);
    }
    
    int trainIndex;
    public void train() {
      trainIndex++;
      if (trainIndex >= images.length)
        trainIndex = 0;
      
      int[] pixels = images[trainIndex];
//      int label = labels[trainIndex];
      float[] realData = pixelsToFloat(pixels);
      
      float[] encoded = encoder.predict(NumpyArray.of(realData)).transpose().data[0];
      float[] decoded = decoder.predict(NumpyArray.of(encoded)).transpose().data[0];
      
      boolean order = Math.random() > 0.5d;
      float[] combined = new float[realData.length * 2];
      System.arraycopy(order ? realData : decoded, 0, combined, 0, imageResolution * imageResolution);
      System.arraycopy(order ? decoded : realData, 0, combined, imageResolution * imageResolution, imageResolution * imageResolution);
      float discriminated = discriminator.predict(NumpyArray.of(combined)).data[0][0];
      
      { // train generator
        NumpyArray grad = discriminator.backwardWithoutTrain(
          discriminatorLoss,
          NumpyArray.of(discriminated),
          NumpyArray.of(order ? 0 : 1),
          learningRate
        );
        
        float[] data = grad.transpose().data[0];
        float[] truncated = new float[imageResolution * imageResolution];
        System.arraycopy(data, order ? 0 : imageResolution * imageResolution, truncated, 0, truncated.length);
        
        grad = NumpyArray.of(truncated).multiply(-1);
        
        grad = decoder.trainWithoutPredict(
          decoderLoss,
          NumpyArray.of(decoded),
          grad,
          learningRate
        );
        grad = grad.multiply(-1);
        
        encoder.trainWithoutPredict(
          encoderLoss,
          NumpyArray.of(encoded),
          grad,
          learningRate
        );
      }
      
      { // train discriminator
        discriminator.trainSingle(
          discriminatorLoss,
          combined,
          new float[] { order ? 0 : 1 },
          learningRate * 0.001f
        );
  
        // train on reverse
        combined = new float[decoded.length * 2];
        System.arraycopy(order ? decoded : realData, 0, combined, 0, imageResolution * imageResolution);
        System.arraycopy(order ? realData : decoded, 0, combined, imageResolution * imageResolution, imageResolution * imageResolution);
        discriminator.trainSingle(
          discriminatorLoss,
          combined,
          new float[] { order ? 1 : 0 },
          learningRate * 0.001f
        );
      }
    }
    
    int thinkIndex;
    int thinkLabel;
    
    void think() {
      thinkIndex++;
      thinkLabel++;
      
      if (thinkLabel > 9)
        thinkLabel = 0;
      thinkIndex %= images.length;
    }
    
    public void paint(Graphics graphics) {
      super.paint(graphics);
      if (thinkIndex == 0)
        return;
      
      int[] pixels = images[thinkIndex];
      float[] encoderInput = pixelsToFloat(pixels);
      float[] encoded = encoder.predictThreadSafe(encoderInput).transpose().data[0];
      float[] decoded = decoder.predictThreadSafe(encoded).transpose().data[0];
      
      int[] decodedPixels = floatToPixels(decoded);
      
      drawPixels(graphics, pixels, 0, 0, getWidth() / 2, getHeight() / 2);
      drawPixels(graphics, decodedPixels, getWidth() / 2, 0, getWidth() / 2, getHeight() / 2);
  
      Graphics2D graphics2D = (Graphics2D) graphics;
      graphics2D.setStroke(new BasicStroke(2f));
  
      drawLines(graphics, data1, 0, getHeight() / 2, getWidth(), getHeight() / 2, Color.red);
      drawLines(graphics, data2, 0, getHeight() / 2, getWidth(), getHeight() / 2, Color.green);
    }
    
    void drawLines(Graphics graphics, List<Float> dataList, float startX, float startY, float width, float height, Color color) {
      float lastX = -1;
      float lastY = -1;
      graphics.setColor(color);
      for (int i = 0; i < dataList.size(); i++) {
        float x = startX + width / dataList.size() * i;
        float y = startY + dataList.get(i) * height;
        
        if(lastX != -1 && lastY != -1) {
          graphics.drawLine(
            (int) x,
            (int) y,
            (int) lastX,
            (int) lastY
          );
        }
        
        lastX = x;
        lastY = y;
      }
    }
    
    void drawPixels(Graphics graphics, int[] pixels, int x, int y, int width, int height) {
      BufferedImage bufferedImage = new BufferedImage(imageResolution, imageResolution, BufferedImage.TYPE_INT_RGB);
      
      for (int i = 0; i < imageResolution * imageResolution; i++) {
        int brightness = pixels[i];
        bufferedImage.setRGB(
          i % imageResolution,
          i / imageResolution,
          new Color(brightness, brightness, brightness).getRGB()
        );
      }
      
      graphics.drawImage(
        bufferedImage,
        x, y,
        width, height,
        null
      );
    }
    
    static int[] downScale(int[] pixels, int toResolution) {
      int[] resized = new int[toResolution * toResolution];
      
      for (int i = 0; i < pixels.length; i++) {
        int x = i % 28;
        int y = i / 28;
        
        x /= 28d / toResolution;
        y /= 28d / toResolution;
        int i2 = x + y * toResolution;
        
        resized[i2] += pixels[i];
      }
      
      float divide = (float) pixels.length / resized.length;
      for (int i = 0; i < resized.length; i++) {
        resized[i] = (int) (resized[i] / divide);
        resized[i] = Math.min(Math.max(resized[i], 0), 255);
      }
      
      return resized;
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
  }
}