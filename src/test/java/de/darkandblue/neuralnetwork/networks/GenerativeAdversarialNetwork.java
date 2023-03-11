package de.darkandblue.neuralnetwork.networks;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.lossfunction.BinaryCrossEntropy;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.lossfunction.MeanSquareError;
import de.darkandblue.neuralnetwork.math.NumpyArray;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class GenerativeAdversarialNetwork extends JFrame {
  public static void main(String[] args) {
    new GenerativeAdversarialNetwork();
  }
  
  Scene scene;
  
  public GenerativeAdversarialNetwork() {
    setTitle("GenerativeAdversarialNetwork");
    
    scene = new Scene();
    add(scene);
    
    setSize(1000, 500);
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
    NeuralNetwork discriminator;
    NeuralNetwork generator;
    int imageResolution = 28;
    int[][] images;
    int[] labels;
    int noiseCount = 64;
    LossFunction generatorLoss = new MeanSquareError();
    LossFunction discriminatorLoss = new BinaryCrossEntropy();
    double learningRate = 0.2;
    
    public Scene() {
      images = MnistLoader.readImages().stream().toArray(int[][]::new);
      if (imageResolution != 28)
        for (int i = 0; i < images.length; i++)
          images[i] = downScale(images[i], imageResolution);
      labels = MnistLoader.readLabels().stream().mapToInt(i -> i).toArray();
      
      generator = new NetworkBuilder()
        .layer.dense(noiseCount + 10, 256)
        .activation.sigmoid()
        .layer.dense(256, imageResolution * imageResolution)
        .activation.sigmoid()
        .build();
      
      discriminator = new NetworkBuilder()
        .layer.dense(imageResolution * imageResolution + 10, 128)
        .activation.sigmoid()
        .layer.dense(128, 1)
        .activation.sigmoid()
        .build();
    }
    
    int trainIndex;
    
    public void train() {
      trainIndex++;
      if (trainIndex >= images.length)
        trainIndex = 0;
      
      double[] noiseData = new double[noiseCount + 10];
      for (int i = 0; i < noiseCount; i++)
        noiseData[i] = Math.random();
      
      int fakeLabel = (int) (Math.random() * 10);
      noiseData[noiseCount + fakeLabel] = 1;
      
      double[] generatedData = generator.predict(NumpyArray.of(noiseData)).transpose().data[0];
      double[] discriminatorInput = new double[generatedData.length + 10];
      System.arraycopy(generatedData, 0, discriminatorInput, 0, generatedData.length);
      discriminatorInput[generatedData.length + fakeLabel] = 1;
      double discriminated = discriminator.predict(NumpyArray.of(discriminatorInput)).data[0][0];
      
      // train generator
      NumpyArray grad = discriminator.backwardWithoutTrain(
        discriminatorLoss,
        NumpyArray.of(discriminated),
        NumpyArray.of(0),
        learningRate
      );
      double[] data = grad.transpose().data[0];
      double[] truncated = new double[imageResolution * imageResolution];
      System.arraycopy(data, 0, truncated, 0, truncated.length);
      grad = NumpyArray.of(truncated).multiplyScalar(-1);
      
      generator.trainWithoutPredict(
        generatorLoss,
        NumpyArray.of(generatedData),
        grad,
        learningRate
      );
      
      // train discriminator
      int[] pixels = images[trainIndex];
      int label = labels[trainIndex];
      double[] realData = pixelsToDouble(pixels);
//      discriminatorInput = new double[realData.length + 10];
      System.arraycopy(realData, 0, discriminatorInput, 0, realData.length);
      discriminatorInput[realData.length + label] = 1;
      
      discriminator.trainSingle(
        discriminatorLoss,
        discriminatorInput,
        new double[] { 0 },
        learningRate
      );

//      discriminatorInput = new double[generatedData.length + 10];
      System.arraycopy(generatedData, 0, discriminatorInput, 0, generatedData.length);
      discriminatorInput[generatedData.length + fakeLabel] = 1;
      discriminator.trainSingle(
        discriminatorLoss,
        discriminatorInput,
        new double[] { 1 },
        learningRate
      );
    }
    
    int thinkIndex;
    int thinkLabel;
    double[] noiseThink = new double[noiseCount];
    
    void think() {
      thinkIndex++;
      thinkLabel++;
      
      if (thinkLabel > 9)
        thinkLabel = 0;
      thinkIndex %= images.length;
      
      for (int i = 0; i < noiseCount; i++)
        noiseThink[i] = Math.random();
    }
    
    public void paint(Graphics graphics) {
      if (thinkIndex == 0)
        return;
      
      double[] thinkInput = new double[noiseCount + 10];
      System.arraycopy(noiseThink, 0, thinkInput, 0, noiseThink.length);
      thinkInput[noiseCount + thinkLabel] = 1;
      double[] generatedData = generator.predictThreadSafe(thinkInput).transpose().data[0];
      int[] pixels = doubleToPixels(generatedData);
      
      BufferedImage bufferedImage = new BufferedImage(imageResolution, imageResolution, BufferedImage.TYPE_INT_RGB);
      
      for (int i = 0; i < imageResolution * imageResolution; i++) {
        int brightness = pixels[i];
        bufferedImage.setRGB(
          i / imageResolution,
          i % imageResolution,
          new Color(brightness, brightness, brightness).getRGB()
        );
      }
      
      graphics.drawImage(
        bufferedImage,
        0, 0,
        getWidth() / 2, getHeight(),
        null
      );
    }
    
    static int[] downScale(int[] pixels, int to) {
      int[] resized = new int[to * to];
      
      for (int i = 0; i < pixels.length; i++) {
        int x = i % 28;
        int y = i / 28;
        
        x /= (double) 28 / to;
        y /= (double) 28 / to;
        int i2 = x + y * to;
        
        resized[i2] += pixels[i];
      }
      
      double divide = (double) pixels.length / resized.length;
      for (int i = 0; i < resized.length; i++) {
        resized[i] = (int) (resized[i] / divide);
        resized[i] = Math.min(Math.max(resized[i], 0), 255);
      }
      
      return resized;
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
  }
}