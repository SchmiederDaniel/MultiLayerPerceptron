package neuralnetwork.models;

import models.dataset.MNISTLoader;
import oldneuralnetwork.NetworkBuilder;
import oldneuralnetwork.NeuralNetwork;
import oldneuralnetwork.lossfunction.MeanSquareError;
import oldneuralnetwork.NumpyArray;

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
      .layer.dense(28 * 28, 40)
      .activation.sigmoid()
      .layer.dense(40, 10)
      .activation.sigmoid()
      .build();
    List<int[]> imageList;
    List<Integer> labelList;
    
    public Scene() throws IOException {
      imageList = MNISTLoader.trainData();
      labelList = MNISTLoader.trainLabels();
      
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
      
      float[] x = pixelsTofloat(pixels);
      float[] y = new float[10];
      y[label] = 1;
      
      neuralNetwork.trainSingle(lossFunction, x, y, 0.1f, false);
    }
    
    float[] pixelsTofloat(int[] pixels) {
      float[] output = new float[pixels.length];
      for (int i = 0; i < pixels.length; i++) {
        output[i] = pixels[i] / 255f;
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
      
      float[] x = pixelsTofloat(pixels);
      NumpyArray predict = neuralNetwork.predict(x);
      float[] y = predict.transpose().data[0];
      
      String out = "";
      for (int i = 0; i < y.length; i++) {
        out += i + "=" + (int) (y[i] * 100d) + "%  ";
      }
      
      graphics.setFont(font);
      graphics.drawString(out, 10, getWidth() + 20);
    }
  }
}