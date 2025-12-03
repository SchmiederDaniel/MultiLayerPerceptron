package neuralnetwork.models;

import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.layer.Dense;
import neuralnetwork.layer.Layer;
import neuralnetwork.lossfunction.MeanSquareError;
import neuralnetwork.math.NumpyArray;
import neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class MNIST extends JFrame {
  public static void main(String[] args) throws IOException {
    new MNIST();
  }
  
  public MNIST() throws IOException {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setSize(1600, 900);
    add(new Scene());
    
    setVisible(true);
    setLocationRelativeTo(null);
  }
  
  class Scene extends JPanel {
    MeanSquareError lossFunction = new MeanSquareError();
    NeuralNetwork neuralNetwork = new NetworkBuilder()
      .layer.dense(28 * 28, 20)
      .activation.sigmoid()
      .layer.dense(20, 10)
      .activation.sigmoid()
      .build();
    List<int[]> imageList;
    List<Integer> labelList;
    List<int[]> testImageList;
    List<Integer> testLabelList;
    
    public Scene() {
      imageList = MnistLoader.readImages();
      labelList = MnistLoader.readLabels();
      
      testImageList = MnistLoader.readTestImages();
      testLabelList = MnistLoader.readTestLabels();
      
      displayImagePixels = testImageList.get(paintIndex);
      
      startAsyncThreads();
    }
    
    void startAsyncThreads() {
      // generating new display image
      new Thread(() -> {
        while (true) {
          try {
            Thread.sleep(2500);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
          paintIndex++;
          paintIndex %= testImageList.size();
          displayImagePixels = testImageList.get(paintIndex);
        }
      }).start();
      
      // rendering
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
      
      // testing
      new Thread(() -> {
        while (true) {
          test();
        }
      }).start();
      
      // training
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
      int label = labelList.get(trainIndex);
      
      float[] x = pixelsTofloat(pixels);
      float[] y = new float[10];
      y[label] = 1;
      
      neuralNetwork.trainSingle(lossFunction, x, y, 0.3f, false);
      
      for (Layer layer : neuralNetwork.layerArray) {
        if (layer instanceof Dense) {
          Dense dense = ((Dense) layer);
          if(trainIndex % 1000 == 0) { // remove faded out graidents
            for (int rowIndex = 0; rowIndex < dense.weights.rows(); rowIndex++) {
              float[] row = dense.weights.data[rowIndex];
              for (int colIndex = 0; colIndex < dense.weights.cols(); colIndex++) {
                float v = row[colIndex];
                if(Math.abs(v) <= 0.0001) {
                  
                }
                dense.weights.data[rowIndex][colIndex] = v;
              }
            }
          }

          // make the gradient fade out
          for (int rowIndex = 0; rowIndex < dense.weights.rows(); rowIndex++) {
            for (int colIndex = 0; colIndex < dense.weights.cols(); colIndex++) {
              float v = dense.weights.data[rowIndex][colIndex];
              float strength = 0.00001f;
              if (v > 0)
                v -= strength;
              else
                v += strength;
              dense.weights.data[rowIndex][colIndex] = v;
            }
          }
        }
      }
    }
    
    float[] pixelsTofloat(int[] pixels) {
      float[] output = new float[pixels.length];
      for (int i = 0; i < pixels.length; i++) {
        output[i] = pixels[i] / 255f;
      }
      return output;
    }
    
    List<Float> testingErrorList = new ArrayList<>();
    
    void test() {
      float sum = 0;
      for (int i = 0; i < 1000; i++) {
        int[] pixels = testImageList.get(i);
        int label = testLabelList.get(i);
        float[] x = pixelsTofloat(pixels);
        
        float[] y = new float[10];
        y[label] = 1;
        
        NumpyArray output = neuralNetwork.predictThreadSafe(x);
        float error = lossFunction.loss(NumpyArray.of(y), output);
        sum += 1 - error;
      }
      float avg = sum / 1000f;
      
      testingErrorList.add(avg);
      if (testingErrorList.size() > 500)
        testingErrorList.remove(0);
    }
    
    DecimalFormat decimalFormat = new DecimalFormat("00.00");
    Font font = new Font("Arial", Font.BOLD, 14);
    int paintIndex;
    
    int[] displayImagePixels;
    
    float highestValueOfArray(float[] array) {
      float max = 0;
      for (float a : array) {
        if (a > max)
          max = a;
      }
      return max;
    }
    
    public void paint(Graphics graphics) {
      super.paint(graphics);
      
      BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
      int[] pixelsOfImage = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
      for (int i = 0; i < pixelsOfImage.length; i++) {
        int brightness = displayImagePixels[i];
        pixelsOfImage[i] = new Color(brightness, brightness, brightness).getRGB();
      }
      graphics.drawImage(image, 0, 0, getWidth() / 2, getHeight(), null);
      
      float[] x = pixelsTofloat(displayImagePixels);
      NumpyArray predict = neuralNetwork.predictThreadSafe(x);
      float[] y = predict.transpose().data[0];
      
      float highestValue = highestValueOfArray(y);
      graphics.setColor(Color.white);
      graphics.setFont(font);
      int label = testLabelList.get(paintIndex);
      for (int i = 0; i < y.length; i++) {
        if (y[i] == highestValue) {
          if (i == label)
            graphics.setColor(Color.green);
          else
            graphics.setColor(Color.red);
        } else {
          graphics.setColor(Color.white);
        }
        graphics.drawString(i + "=" + (int) (y[i] * 100d) + "%", (int) (10 + i / 10d * getWidth() / 2), getHeight() - 20);
      }
      
      drawLine(
        graphics,
        testingErrorList,
        getWidth() / 2,
        0,
        getWidth() / 2,
        getHeight(),
        Color.red
      );
      
      if (testingErrorList.size() > 0) {
        graphics.setColor(Color.black);
        graphics.setFont(font);
        graphics.drawString(
          "Error: " + ((1d - testingErrorList.get(testingErrorList.size() - 1)) * 10),
          getWidth() / 2 + 10,
          getHeight() - 20
        );
      }
      
      int posY = 0;
      float size = 5; // size of weight in pixel
      for (Layer layer : neuralNetwork.layerArray) {
        if (layer instanceof Dense) {
          Dense dense = ((Dense) layer);
          int posX = 0;
          for (float[] a : dense.weights.data) {
            for (float b : a) {
              b = Math.max(Math.min(b, 1), -1);
              int bightness = (int) ((b + 1d) * 127d);
              graphics.setColor(new Color(bightness, bightness, bightness));
              while (posX > getWidth() / 2) {
                posX = 0;
                posY += size;
              }
              graphics.fillRect(getWidth() / 2 + posX, posY, (int) size, (int) size);
              posX += size;
            }
            posY += size * 2;
            posX = 0;
          }
          posY += 20;
        }
      }
    }
    
    void drawLine(Graphics graphics, List<Float> list, int x, int y, int width, int height, Color color) {
      int lastX2 = -1;
      int lastY2 = -1;
      graphics.setColor(color);
      Graphics2D graphics2D = (Graphics2D) graphics;
      graphics2D.setStroke(new BasicStroke(2f));
      
      for (int i = 0; i < list.size(); i++) {
        float element = list.get(i);
        
        int x2 = (int) (x + (float) width / list.size() * i);
        int y2 = (int) (y + (float) height / 1 * element);
        
        if (lastX2 != -1) {
          graphics.drawLine(lastX2, lastY2, x2, y2);
        }
        
        lastX2 = x2;
        lastY2 = y2;
      }
    }
  }
}