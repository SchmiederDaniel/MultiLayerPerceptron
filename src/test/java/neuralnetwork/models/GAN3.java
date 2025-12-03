package neuralnetwork.models;

import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.lossfunction.BinaryCrossEntropy;
import neuralnetwork.lossfunction.LossFunction;
import neuralnetwork.math.NumpyArray;
import neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Stack;

public class GAN3 extends JFrame {
  public static void main(String[] args) {
    new GAN3();
  }
  
  Scene scene;
  
  public GAN3() {
    setTitle("GAN3");
    
    scene = new Scene();
    add(scene);
    
    setSize(600, 300);
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
    int noiseCount = 10;
    //    LossFunction generatorLoss = new AbsoluteLoss();
    LossFunction discriminatorLoss = new BinaryCrossEntropy();
    float learningRate = 0.05f;
    float discriminatorLearningRate = learningRate * 0.25f;
    
    float[] getNoiseData(int length) {
      float[] result = new float[length];
           
      for (int i = 0; i < length; i++) {
//        result[i] = (float) Math.random() * 2f - 1f; // for leaky relu
        result[i] = (Float.MAX_VALUE - Float.MAX_VALUE / 2) * 2f; // for sigmoid
      }
      return result;
    }
    
    public Scene() {
      images = MnistLoader.readImages().stream().toArray(int[][]::new);
      if (imageResolution != 28)
        for (int i = 0; i < images.length; i++)
          images[i] = downScale(images[i], imageResolution);
      labels = MnistLoader.readLabels().stream().mapToInt(i -> i).toArray();
      
      generator = new NetworkBuilder()
          .distribution.xavier()
//          .distribution.customDistribution(0.001f, 0.002f, -0.05f, 0.05f)
          .layer.dense(noiseCount, 50)
          .activation.sigmoid()
          .layer.dropOut(0.25f)
          .layer.dense(50, 200)
          .activation.sigmoid()
          .layer.dropOut(0.25f)
          .layer.dense(200, 200)
          .activation.sigmoid()
          .layer.dense(200, imageResolution * imageResolution)
          .activation.sigmoid()
          .build();
      
      discriminator = new NetworkBuilder()
          .distribution.xavier()
//          .layer.dropOut(0.8f)
          .layer.dense(imageResolution * imageResolution, 80)
          .activation.sigmoid()
          .layer.dense(80, 80)
          .activation.sigmoid()
          .layer.dense(80, 50)
          .activation.sigmoid()
          .layer.dense(50, 1)
          .activation.sigmoid()
          .build();
    }
    
    int trainIndex;
    Stack<Float> lastErrors = new Stack<>();
    
    public void train() {
      trainIndex++;
      if (trainIndex >= images.length)
        trainIndex = 0;
      
      float[] noiseData = getNoiseData(noiseCount);
      
      float[] generatedData = generator.predict(NumpyArray.of(noiseData)).transpose().data[0];
      float discriminated = discriminator.predict(NumpyArray.of(generatedData)).data[0][0];
      
      // train generator
      NumpyArray grad = discriminator.backwardWithoutTrain(
          discriminatorLoss,
          NumpyArray.of(discriminated),
          NumpyArray.of(0),
          learningRate
      );
      float[] data = grad.transpose().data[0];
      grad = NumpyArray.of(data);
      
      generator.continueBackpropagation(
          grad,
          learningRate
      );
      
      // train discriminator
      int[] pixels = images[trainIndex];
      float[] realData = pixelsToFloat(pixels);
      
      discriminator.trainSingle(
          discriminatorLoss,
          realData,
          new float[]{0},
          discriminatorLearningRate
      );
      
      float error = discriminator.trainSingle(
          discriminatorLoss,
          generatedData,
          new float[]{1},
          discriminatorLearningRate
      );
      lastErrors.add(error);
    }
    
    int thinkIndex;
    float[] noiseThink = new float[noiseCount];
    
    void think() {
      thinkIndex++;
      thinkIndex %= images.length;
      
      noiseThink = getNoiseData(noiseCount);
      
      Stack<Float> copyLoss = (Stack<Float>) lastErrors.clone();
      if (copyLoss.size() > 1) {
        float sum = 0;
        for (float loss : copyLoss)
          sum += loss;
        System.out.println("Disc-loss: " + String.format("%.4f", sum / copyLoss.size()) + " loss-count: " + copyLoss.size());
        lastErrors.clear();
      }
    }
    
    public void paint(Graphics graphics) {
      if (thinkIndex == 0)
        return;
      
      float[] generatedData = generator.predictThreadSafe(noiseThink).transpose().data[0];
      int[] pixels = floatToPixels(generatedData);
      
      BufferedImage bufferedImage = new BufferedImage(imageResolution, imageResolution, BufferedImage.TYPE_INT_RGB);
      
      for (int i = 0; i < imageResolution * imageResolution; i++) {
        int brightness = pixels[i];
        brightness = Math.max(Math.min(brightness, 255), 0);
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
        
        x /= (int) ((float) 28 / to);
        y /= (int) ((float) 28 / to);
        int i2 = x + y * to;
        
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