package de.darkandblue.neuralnetwork.models;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.lossfunction.*;
import de.darkandblue.neuralnetwork.math.NumpyArray;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.BufferedImage;
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
        
        TODO: Idea for corrected training: Make one forward pass and another one. than train the neural network on the
        loss between two steps (Maybe adding a new value to the input which defines the denoise strength could help) 
   */
  Scene scene;
  static JSlider sliderViewSteps = new JSlider(1, Scene.MAX_STEP_SIZE - 1, Scene.MAX_STEP_SIZE / 2);
  JSlider sliderLearningRate = new JSlider(0, 100000, (int) (Scene.learningRate * 100000d * 4d));
  
  public Diffusion() {
    setLayout(null);
    scene = new Scene();
    add(scene);
    setBackground(new Color(120, 120, 120));
    
    JLabel labelNoise = new JLabel("Strength of noising before denoising (0.5):");
    add(labelNoise);
    add(sliderViewSteps);
    sliderViewSteps.addChangeListener(e -> {
      labelNoise.setText("Strength of noising before denoising (" + Math.round((float) sliderViewSteps.getValue() / Scene.MAX_STEP_SIZE * 100d) / 100d + "):");
    });
    
    JLabel labelLR = new JLabel("LearningRate: " + Math.round(Scene.learningRate * 100d) / 100d);
    add(labelLR);
    add(sliderLearningRate);
    sliderLearningRate.addChangeListener(e -> {
      Scene.learningRate = (float) sliderLearningRate.getValue() / sliderLearningRate.getMaximum() / 4f;
      labelLR.setText("LearningRate: " + Math.round(Scene.learningRate * 10000d) / 10000d);
    });
    
    addComponentListener(new ComponentAdapter() {
      public void componentResized(ComponentEvent e) {
        labelNoise.setSize(getWidth(), 20);
        labelNoise.setLocation(10, getHeight() - 150);
        
        sliderViewSteps.setSize(getWidth() - 20, 30);
        sliderViewSteps.setLocation(0, labelNoise.getY() + labelNoise.getHeight());
        
        labelLR.setSize(getWidth(), 20);
        labelLR.setLocation(10, sliderViewSteps.getY() + sliderViewSteps.getHeight());
        
        sliderLearningRate.setSize(getWidth() - 20, 30);
        sliderLearningRate.setLocation(0, labelLR.getY() + labelLR.getHeight());
        
        scene.setSize(getWidth() - 4, getWidth());
      }
    });
    
    setSize(800, 950);
    setVisible(true);
    setLocationRelativeTo(null);
    
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    
  }
  
  class Scene extends JPanel {
    static int imageResolution = 28;
    int[][] images;
    static int[][] testImages;
    int[] labels;
    int[] testLabels;
    
    private static final int STEP_INDEX_COUNT = 10;
    private static final int MAX_STEP_SIZE = 800;
    static float learningRate = 0.05f;
    LossFunction lossFunction = new AbsoluteLoss();
    static NeuralNetwork neuralNetwork = new NetworkBuilder()
      .distribution.customDistribution(0.01f, 0.05f, -0.52f, 0.52f)
      .layer.dense(imageResolution * imageResolution + STEP_INDEX_COUNT, 140)
      .activation.sigmoid()
      .layer.dense(140, 140)
      .activation.sigmoid()
      .layer.dense(140, imageResolution * imageResolution)
      .activation.sigmoid()
      .build();
    static float thinkStepSize = 1f; // 0.5d = best
    
    public Scene() {
      images = MnistLoader.readImages().stream().toArray(int[][]::new);
      if (imageResolution != 28)
        for (int i = 0; i < images.length; i++)
          images[i] = downScale(images[i], imageResolution);
      labels = MnistLoader.readLabels().stream().mapToInt(i -> i).toArray();
      
      testImages = MnistLoader.readTestImages().stream().toArray(int[][]::new);
      testLabels = MnistLoader.readTestLabels().stream().mapToInt(i -> i).toArray();
      
      startAsyncThreads();
    }
    
    void startAsyncThreads() {
      new Thread(() -> {
        while (true) {
          this.bufferedImage = scene.think(Math.max(getWidth(), 1), Math.max(getHeight(), 1));
        }
      }).start();
      
      new Thread(() -> {
        while (true) {
          thinkIndex++;
          thinkIndex %= testImages.length;
          try {
            Thread.sleep(1800);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        }
      }).start();
      
      new Thread(() -> {
        while (true) {
          train();
        }
      }).start();
      
      new Thread(() -> {
        while (true) {
          repaint();
          try {
            Thread.sleep(32);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        }
      }).start();
    }
    
    int trainIndex;
    
    public void train() {
      trainIndex++;
      if (trainIndex >= images.length)
        trainIndex = 0;
      int i = ThreadLocalRandom.current().nextInt(MAX_STEP_SIZE);
      
      int[] pixels = images[trainIndex];
//      int label = labels[trainIndex];
      float[] realData = pixelsTofloat(pixels);
      
      float[] input = addNoise(trainIndex, realData, convertNoiseStrength((float) i / MAX_STEP_SIZE));
      float[] target = addNoise(trainIndex, realData, convertNoiseStrength((i - 1f) / MAX_STEP_SIZE));
      target = subtract(input, target);
      target = multiply(target, MAX_STEP_SIZE * 2f); // TODO: find out why do I need to multiply by 2 instead of deviding (usally it should take more space for -1 and +1 values inside 0-1)
      target = add(target, 0.5f);
      
      float[] state = new float[STEP_INDEX_COUNT];
      state[(int) ((float) i / MAX_STEP_SIZE * STEP_INDEX_COUNT)] = 1;
      float[] input2 = addValuesToArray(input, state);
      
      neuralNetwork.trainSingle(lossFunction, input2, target, learningRate);

//      float[] output = neuralNetwork.trainSingle(lossFunction, input2, target, learningRate).transpose().data[0];
//      output = subtract(output, 0.5);
//      output = devide(output, MAX_STEP_SIZE / 2d);
//      input = subtract(input, output);
//      input = minmax(input, 0, 1);
//  
//      target = addNoise(trainIndex, realData, convertNoiseStrength((i - 2d) / MAX_STEP_SIZE));
//      target = subtract(input, target);
//      target = multiply(target, MAX_STEP_SIZE * 2d);
//      target = add(target, 0.5);
//  
//      input = addValuesToArray(input, state);
//      neuralNetwork.trainSingle(lossFunction, input, target, learningRate / 2d);
    }
    
    static float[] addNoise(int seed, float[] array, float strength) {
      float[] output = new float[array.length];
      Random random = new Random(seed);
      
      for (int i = 0; i < array.length; i++) {
        float value = array[i];
        float diffrence = random.nextFloat() - value;
        output[i] = value + diffrence * strength;
      }
      return output;
    }
    
    static float convertNoiseStrength(float x) {
      return x - x * x + x;
//      return x * x;
//      return x;
    }
    
    static int thinkIndex;
    
    BufferedImage bufferedImage = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
    
    static BufferedImage think(int width, int height) {
      BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
      Graphics2D graphics = bufferedImage.createGraphics();
      
      float[] pixels = pixelsTofloat(testImages[thinkIndex]);
      float[] noisy = addNoise(thinkIndex, pixels, (float) sliderViewSteps.getValue() / MAX_STEP_SIZE);
      
      int[] decodedPixels = floatToPixels(noisy);
      drawPixels(graphics, decodedPixels, 0, height / 5 * 4, width / 5, height / 5);
      decodedPixels = floatToPixels(pixels);
      drawPixels(graphics, decodedPixels, height / 5, height / 5 * 4, width / 5, height / 5);
      
      float imageSize = width / 5f;
      
      float[] noisyOriginal = noisy;
      float imageCounter = 0;
      int counter = 0;
      for (float i = sliderViewSteps.getValue(); i >= 0; i -= thinkStepSize) {
        float[] state = new float[STEP_INDEX_COUNT];
        state[(int) (i / MAX_STEP_SIZE * STEP_INDEX_COUNT)] = 1;
        float[] input = addValuesToArray(noisy, state);
        float[] output = neuralNetwork.predictThreadSafe(input).transpose().data[0];
        output = subtract(output, 0.5f);
        output = devide(output, MAX_STEP_SIZE / 2f); // Needs to be amplified. For explanation see in training
        noisy = subtract(noisy, output);
        noisy = minmax(noisy, 0, 1);
        
        int i2 = (int) (sliderViewSteps.getValue() - i);
        if (((float) i2 / sliderViewSteps.getValue() * 20d) - imageCounter > 1 || i2 == sliderViewSteps.getValue() - 1) {
          decodedPixels = floatToPixels(noisy);
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
          imageCounter = (float) i2 / sliderViewSteps.getValue() * 20f;
          counter++;
        }
      }
      drawPixels(graphics, floatToPixels(noisy), height / 5 * 2, height / 5 * 4, width / 5, height / 5);
      
      float[] difference = NumpyArray.of(noisyOriginal).subtract(NumpyArray.of(noisy)).add(NumpyArray.of(0.5f)).transpose().data[0];
      drawPixels(graphics, floatToPixels(difference), height / 5 * 3, height / 5 * 4, width / 5, height / 5);
      
      return bufferedImage;
    }
    
    public void paint(Graphics graphics) {
      super.paint(graphics);
      if (thinkIndex == 0)
        return;

//      int label = testLabels[thinkIndex];
      setTitle("training steps: " + trainIndex + " learning rate: " + learningRate + " diffusion step size: " + MAX_STEP_SIZE);
      graphics.drawImage(bufferedImage, 0, 0, getWidth(), getHeight(), null);
    }
    
    static float[] addValuesToArray(float[] array, float... values) {
      float[] output = new float[array.length + values.length];
      System.arraycopy(array, 0, output, 0, array.length);
      System.arraycopy(values, 0, output, array.length, values.length);
      return output;
    }
    
    static void drawPixels(Graphics graphics, int[] pixels, int x, int y, int width, int height) {
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
      
      float divide = (float) pixels.length / resized.length;
      for (int i = 0; i < resized.length; i++) {
        resized[i] = (int) (resized[i] / divide);
        resized[i] = Math.min(Math.max(resized[i], 0), 255);
      }
      
      return resized;
    }
    
    static float[] pixelsTofloat(int[] pixels) {
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
    
    static float[] subtract(float[] value1, float[] value2) {
      if (value1.length != value2.length)
        throw new RuntimeException("length doesnt match");
      float[] output = new float[value1.length];
      for (int i = 0; i < value1.length; i++) {
        output[i] = value1[i] - value2[i];
      }
      return output;
    }
    
    static float[] minmax(float[] array, float min, float max) {
      float[] output = new float[array.length];
      for (int i = 0; i < array.length; i++) {
        float value = array[i];
        output[i] = value < min ? min : value;
        output[i] = value > max ? max : value;
      }
      return output;
    }
    
    static float[] multiply(float[] value1, float value2) {
      float[] output = new float[value1.length];
      for (int i = 0; i < value1.length; i++) {
        output[i] = value1[i] * value2;
      }
      return output;
    }
    
    static float[] devide(float[] value1, float value2) {
      float[] output = new float[value1.length];
      for (int i = 0; i < value1.length; i++) {
        output[i] = value1[i] / value2;
      }
      return output;
    }
    
    static float[] subtract(float[] value1, float value2) {
      float[] output = new float[value1.length];
      for (int i = 0; i < value1.length; i++) {
        output[i] = value1[i] - value2;
      }
      return output;
    }
    
    static float[] add(float[] value1, float value2) {
      float[] output = new float[value1.length];
      for (int i = 0; i < value1.length; i++) {
        output[i] = value1[i] + value2;
      }
      return output;
    }
  }
}