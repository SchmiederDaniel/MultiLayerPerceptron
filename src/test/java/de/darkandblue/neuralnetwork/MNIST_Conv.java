package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.lossfunction.MeanSquareError;
import de.darkandblue.neuralnetwork.math.NumpyArray;
import de.darkandblue.neuralnetwork.util.MnistLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;

public class MNIST_Conv extends JFrame {
  public static void main(String[] args) throws IOException {
    new MNIST();
  }
  
  public MNIST_Conv() throws IOException {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setSize(800, 900);
    add(new Scene());
    
    setVisible(true);
    setLocationRelativeTo(null);
  }
  
  class Scene extends JPanel {
    MeanSquareError lossFunction = new MeanSquareError();
    NeuralNetwork neuralNetwork = new NetworkBuilder()
      .dense(28 * 28, 40)
      .sigmoid()
      .dense(40, 10)
      .sigmoid()
      .build();
    List<int[]> imageList;
    List<Integer> labelList;
    
    public Scene() throws IOException {
      imageList = MnistLoader.readImages();
      labelList = MnistLoader.readLabels();
      
      startAsyncThreads();
    }
    
    void startAsyncThreads() {
      // rendering
      new Thread(()->{
        while(true) {
          try {
            Thread.sleep(2500);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
          repaint();
        }
      }).start();
      
      // training
      new Thread(()->{
        while(true) {
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
      
      double[] x = pixelsToDouble(pixels);
      double[] y = new double[10];
      y[label] = 1;
      
      neuralNetwork.trainSingle(lossFunction, x, y, 0.1, false);
    }
    
    double[] pixelsToDouble(int[] pixels) {
      double[] output = new double[pixels.length];
      for (int i = 0; i < pixels.length; i++) {
        output[i] = pixels[i] / 255d;
      }
      return output;
    }
    
    DecimalFormat decimalFormat = new DecimalFormat("00.00");
    Font font = new Font("Consolas", Font.BOLD, 14);
    int paintIndex;
    public void paint(Graphics graphics) {
      super.paint(graphics);
      paintIndex++;
      paintIndex %= 60000;
      int[] pixels = imageList.get(paintIndex);
      
      BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
      int[] pixelsOfImage = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
      for (int i = 0; i < pixelsOfImage.length; i++) {
        pixelsOfImage[i] = new Color(pixels[i], pixels[i], pixels[i]).getRGB();
      }
      graphics.drawImage(image, 0, 0, getWidth(), getWidth(), null);
      
      double[] x = pixelsToDouble(pixels);
      NumpyArray predict = neuralNetwork.predict(x);
      double[] y = predict.transpose().data[0];
      
      String out = "";
      for (int i = 0; i < y.length; i++) {
        out += i + "=" + (int) (y[i] * 100d) + "%  ";
      }
      
      graphics.setFont(font);
      graphics.drawString(out, 10, getWidth() + 20);
    }
  }
}