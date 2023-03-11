package de.darkandblue.neuralnetwork.networks;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.lossfunction.BinaryCrossEntropy;
import de.darkandblue.neuralnetwork.lossfunction.LinearLoss;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.lossfunction.MeanSquareError;
import de.darkandblue.neuralnetwork.math.NumpyArray;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class Diffusion extends JFrame {
  public static void main(String[] args) {
    new Diffusion();
  }
  
  /*
  TODO: Improve the diffusion proccess to make the third preview image represent the negative of the actual number
        or Atleast make it possible to assume/detect which digit is shown in the negative preview.
        
        The Problem is that the diffusion proccess just removes the brightness of the whole image without checking
        which areas of the image are important and shouldn't be denoised. 
   */
  Scene scene;
  JSlider sliderViewSteps = new JSlider(1, Scene.MAX_STEP_SIZE - 1, Scene.MAX_STEP_SIZE / 2);
  
  public Diffusion() {
    setLayout(null);
    scene = new Scene();
    add(scene);
    setBackground(new Color(120, 120, 120));
    
    JLabel label = new JLabel("Strength of noising before denoising (0.5):");
    add(label);
    
    add(sliderViewSteps);
    sliderViewSteps.addChangeListener(e -> {
      label.setText("Strength of noising before denoising (" + Math.round((double) sliderViewSteps.getValue() / Scene.MAX_STEP_SIZE * 100d) / 100d + "):");
    });
    
    addComponentListener(new ComponentAdapter() {
      public void componentResized(ComponentEvent e) {
        label.setLocation(0, getHeight() - 150);
        sliderViewSteps.setLocation(0, getHeight() - 100);
        sliderViewSteps.setSize(getWidth(), 50);
        scene.setSize(getWidth() - 4, getHeight() - 150);
        label.setSize(getWidth(), 50);
      }
    });
    
    setSize(800, 950);
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
      }
    }).start();
  }
  
  class Scene extends JPanel {
    NeuralNetwork neuralNetwork;
    int imageResolution = 28;
    int[][] images;
    int[] labels;
    LossFunction lossFunction = new LinearLoss();
    double learningRate = 0.05;
    private static final int MAX_STEP_SIZE = 500;
    
    public Scene() {
      images = MnistLoader.readImages().stream().toArray(int[][]::new);
      if (imageResolution != 28)
        for (int i = 0; i < images.length; i++)
          images[i] = downScale(images[i], imageResolution);
      labels = MnistLoader.readLabels().stream().mapToInt(i -> i).toArray();
      
      neuralNetwork = new NetworkBuilder()
        .layer.dense(imageResolution * imageResolution + MAX_STEP_SIZE, 300)
        .activation.sigmoid()
        .layer.dense(300, 150)
        .activation.sigmoid()
        .layer.dense(150, 300)
        .activation.sigmoid()
        .layer.dense(300, imageResolution * imageResolution)
        .activation.sigmoid()
        .build();
    }
    
    int trainIndex;
    
    void test() {
      JFrame frame = new JFrame() {
        int index = MAX_STEP_SIZE;
        
        public void paint(Graphics graphics) {
          index++;
          
          double index = this.index % 2 == 0 ? MAX_STEP_SIZE : MAX_STEP_SIZE - 1;
          
          int[] pixels = images[0];
          double[] realData = pixelsToDouble(pixels);
          
          double[] noisy = addNoise(0, realData, index / (double) MAX_STEP_SIZE);
          int[] decodedPixels = doubleToPixels(noisy);
          drawPixels(graphics, decodedPixels, 0, 0, getWidth(), getHeight());
        }
      };
      
      new Thread(() -> {
        while (true) {
          try {
            Thread.sleep(250);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
          frame.repaint();
        }
      }).start();
      frame.setSize(500, 500);
      frame.setVisible(true);
      frame.setLocationRelativeTo(null);
    }
    
    public void train() {
      trainIndex++;
      if (trainIndex >= images.length)
        trainIndex = 0;
      int i = ThreadLocalRandom.current().nextInt(MAX_STEP_SIZE);
      
      int[] pixels = images[trainIndex];
//      int label = labels[trainIndex];
      double[] realData = pixelsToDouble(pixels);
      
      double[] noisy = addNoise(trainIndex, realData, i / (double) MAX_STEP_SIZE);
      double[] lessNoisy = addNoise(trainIndex, realData, (i - 1) / (double) MAX_STEP_SIZE);
      double[] diffrence = subtract(noisy, lessNoisy);
      diffrence = multiply(diffrence, MAX_STEP_SIZE);
      diffrence = add(diffrence, 0.5);
      
      double[] stepIndicator = new double[MAX_STEP_SIZE];
      stepIndicator[i] = 1;
      noisy = addValuesToArray(noisy, stepIndicator);
      
      neuralNetwork.trainSingle(lossFunction, noisy, diffrence, learningRate);
    }
    
    static double[] addNoise(int seed, double[] array, double strength) {
      double[] output = new double[array.length];
      Random random = new Random(seed);
      
      for (int i = 0; i < array.length; i++) {
        double value = array[i];
        double diffrence = random.nextDouble() - value;
        output[i] = value + diffrence * strength;
      }
      
      return output;
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
      setTitle("training steps: " + trainIndex + " learning rate: " + learningRate + " diffusion step size: " + MAX_STEP_SIZE);
      super.paint(graphics);
      if (thinkIndex == 0)
        return;
      
      int[] pixels = images[thinkIndex];
      double[] input = pixelsToDouble(pixels);
      double[] noisy = addNoise(thinkIndex, input, (double) sliderViewSteps.getValue() / MAX_STEP_SIZE);
      
      int[] decodedPixels = doubleToPixels(noisy);
      //int i = MAX_STEP_SIZE / devideStrength - 1; if(; i >= 0; i--)
      drawPixels(graphics, decodedPixels, 0, getHeight() / 5 * 4, getWidth() / 5, getHeight() / 5);
      decodedPixels = doubleToPixels(input);
      drawPixels(graphics, decodedPixels, getHeight() / 5, getHeight() / 5 * 4, getWidth() / 5, getHeight() / 5);
      
      double imageSize = getWidth() / 5;
      
      double[] noisyOriginal = noisy;
      double imageCounter = 0;
      int counter = 0;
      for (int i = 0; i < sliderViewSteps.getValue(); i++) {
        double[] stepIndicator = new double[MAX_STEP_SIZE];
        stepIndicator[sliderViewSteps.getValue() - i] = 1;
        double[] nextInput = addValuesToArray(noisy, stepIndicator);
        double[] output = neuralNetwork.predictThreadSafe(nextInput).transpose().data[0];
        output = subtract(output, 0.5);
        output = devide(output, MAX_STEP_SIZE);
        noisy = subtract(noisy, output);
  
        if(((double) i / sliderViewSteps.getValue() * 20d) - imageCounter > 1 || i == sliderViewSteps.getValue() - 1) {
          decodedPixels = doubleToPixels(noisy);
          int x = counter % 5;
          int y = counter / 5;
          drawPixels(
            graphics,
            decodedPixels,
            (int) (imageSize * x),
            (int) (imageSize * y),
            (int) imageSize,
            (int) imageSize
          );
          imageCounter = (double) i / sliderViewSteps.getValue() * 20d;
          counter++;
        }
      }
      drawPixels(graphics, doubleToPixels(noisy), getHeight() / 5 * 2, getHeight() / 5 * 4, getWidth() / 5, getHeight() / 5);
      
      double[] difference = NumpyArray.of(noisyOriginal).subtract(NumpyArray.of(noisy)).add(NumpyArray.of(0.5)).transpose().data[0];
      drawPixels(graphics, doubleToPixels(difference), getHeight() / 5 * 3, getHeight() / 5 * 4, getWidth() / 5, getHeight() / 5);
      
      Graphics2D graphics2D = (Graphics2D) graphics;
      graphics2D.setStroke(new BasicStroke(2f));
    }
    
    static double[] addValuesToArray(double[] array, double... values) {
      double[] output = new double[array.length + values.length];
      System.arraycopy(array, 0, output, 0, array.length);
      System.arraycopy(values, 0, output, array.length, values.length);
      return output;
    }
    
    void drawPixels(Graphics graphics, int[] pixels, int x, int y, int width, int height) {
      BufferedImage bufferedImage = new BufferedImage(imageResolution, imageResolution, BufferedImage.TYPE_INT_RGB);
      
      for (int i = 0; i < imageResolution * imageResolution; i++) {
        int brightness = Math.max(Math.min(pixels[i], 255), 0);
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
    
    static double[] subtract(double[] value1, double[] value2) {
      if (value1.length != value2.length)
        throw new RuntimeException("length doesnt match");
      double[] output = new double[value1.length];
      for (int i = 0; i < value1.length; i++) {
        output[i] = value1[i] - value2[i];
      }
      return output;
    }
    
    static double[] multiply(double[] value1, double value2) {
      double[] output = new double[value1.length];
      for (int i = 0; i < value1.length; i++) {
        output[i] = value1[i] * value2;
      }
      return output;
    }
    
    static double[] devide(double[] value1, double value2) {
      double[] output = new double[value1.length];
      for (int i = 0; i < value1.length; i++) {
        output[i] = value1[i] / value2;
      }
      return output;
    }
    
    static double[] subtract(double[] value1, double value2) {
      double[] output = new double[value1.length];
      for (int i = 0; i < value1.length; i++) {
        output[i] = value1[i] - value2;
      }
      return output;
    }
    
    static double[] add(double[] value1, double value2) {
      double[] output = new double[value1.length];
      for (int i = 0; i < value1.length; i++) {
        output[i] = value1[i] + value2;
      }
      return output;
    }
  }
}