package de.darkandblue.neuralnetwork;

import de.darkandblue.neuralnetwork.lossfunction.MSE;
import de.darkandblue.neuralnetwork.math.NumpyArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;

public class XOR extends JFrame {
  double[][] X = {
    { 0, 0 },
    { 0, 1 },
    { 1, 0 },
    { 1, 1 },
  };
  double[][] Y = {
    { 0 },
    { 1 },
    { 1 },
    { 0 },
  };
  
  NeuralNetwork neuralNetwork = new NetworkBuilder()
    .dense(2, 3)
    .sigmoid()
    .dense(3, 1)
    .sigmoid()
    .build();
  
  public static void main(String[] args) {
    new XOR();
  }
  
  public XOR() {
    neuralNetwork.train(new MSE(), X, Y, 10000, 0.1, true);
    
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setSize(800, 800);
    JPanel panel = new JPanel() {
      @Override
      public void paint(Graphics graphics) {
        BufferedImage image = new BufferedImage(30, 30, BufferedImage.TYPE_INT_BGR);
        int[] pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        
        for (int xInt = 0; xInt < image.getWidth(); xInt++) {
          for (int yInt = 0; yInt < image.getHeight(); yInt++) {
            double x = (double) xInt / image.getWidth();
            double y = (double) yInt / image.getHeight();
            
            NumpyArray predict = neuralNetwork.predict(x, y);
  
            double brightness = predict.data[0][0];
            int pixel = (int) Math.max(Math.min(brightness * 255d, 255), 0);
            pixels[xInt + yInt * image.getWidth()] = new Color(pixel, pixel, pixel).getRGB();
          }
        }
        
        graphics.drawImage(image, 0, 0, getWidth(), getHeight(), null);
      }
    };
    add(panel);
    
    setVisible(true);
    setLocationRelativeTo(null);
  }
}