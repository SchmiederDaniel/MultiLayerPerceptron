package de.darkandblue.neuralnetwork.benchmark;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.lossfunction.BinaryCrossEntropy;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class Benchmark {
  static final double[][][] trainingData = new double[][][] {
    { { 1, 1 }, { 0 } },
    { { 1, 0 }, { 1 } },
    { { 0, 1 }, { 1 } },
    { { 0, 0 }, { 0 } }
  };
  
  static final LossFunction LOSS_FUNCTION = new BinaryCrossEntropy();
  
  public static void main(String[] args) {
    NeuralNetwork neuralNetwork = new NetworkBuilder()
      .layer.dense(
        new NumpyArray(
          new double[][] {
            { -0.5845703173805659, -0.33456588808097765 },
            { 0.9355118188482414, -0.9877656354684774 }
          }
        ),
        new NumpyArray(new double[][] {
          { 0.4617563814065817 },
          { -0.17983837701559668 }
        })
      )
      .activation.sigmoid()
      .layer.dense(
        new NumpyArray(
          new double[][] {
            { 0.8797307775638197, 0.8943898353263877 },
          }
        ),
        new NumpyArray(new double[][] {
          { 0.9274095940464153 }
        })
      )
      .activation.sigmoid()
      .build();
    
    JFrame frame = new JFrame();
    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    frame.setSize(800, 800);
    double size = 50;
    BufferedImage bufferedImage = new BufferedImage((int) size, (int) size, BufferedImage.TYPE_INT_RGB);
    JPanel panel = new JPanel() {
      public void paint(Graphics graphics) {
        for (int x = 0; x < size; x++) {
          for (int y = 0; y < size; y++) {
            NumpyArray predict = neuralNetwork.predictThreadSafe(
              x / size,
              y / size
            );
            int brightness = (int) (predict.data[0][0] * 255d);
            brightness = Math.min(brightness, 255);
            brightness = Math.max(brightness, 0);
            bufferedImage.setRGB(x, y, new Color(brightness, brightness, brightness).getRGB());
          }
        }
        graphics.drawImage(bufferedImage, 0, 0, getWidth(), getHeight(), null);
      }
    };
    frame.add(panel);
    frame.setVisible(true);
    frame.setLocationRelativeTo(null);
    
    new Thread(() -> {
      int counter = 0;
      boolean stop = false;
      double timeTotal = 0;
      while (!stop) {
        double error = 0;
        double time = 0;
        for (double[][] data : trainingData) {
          long timeStamp = System.nanoTime();
          double[] inputs = data[0];
          double[] targets = data[1];
          
          neuralNetwork.trainSingle(LOSS_FUNCTION, inputs, targets, 0.0001d, false);
          
          time += System.nanoTime() - timeStamp;
          counter++;
          double output = neuralNetwork.predict(NumpyArray.of(inputs)).transpose().data[0][0];
          output = Math.min(output, 1);
          output = Math.max(output, 0);
          error += Math.abs(targets[0] - output);
        }
        if (error < 0.2)
          stop = true;
        timeTotal += time / trainingData.length;
        panel.repaint();
      }
      
      System.out.println("trained " + counter + " times");
      System.out.println("average train time " + timeTotal / counter + " ns");
    }).
      
      start();
  }
}