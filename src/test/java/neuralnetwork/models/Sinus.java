package neuralnetwork.models;

import oldneuralnetwork.NetworkBuilder;
import oldneuralnetwork.NeuralNetwork;
import oldneuralnetwork.lossfunction.LossFunction;
import oldneuralnetwork.lossfunction.MeanSquareError;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.util.ArrayList;
import java.util.Collections;

public class Sinus extends JFrame {
  public static void main(String[] args) {
    new Sinus();
  }
  
  public Sinus() {
    Scene scene = new Scene();
    add(scene);
    
    addComponentListener(new ComponentAdapter() {
      public void componentResized(ComponentEvent e) {
        scene.setSize(getWidth(), getHeight());
      }
    });
    
    setSize(1800, 1000);
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setVisible(true);
    setLocationRelativeTo(null);
  }
  
  class Scene extends JPanel {
    NeuralNetwork neuralNetwork = new NetworkBuilder()
      .layer.dense(1, 80)
      .activation.tanh()
      .layer.dense(80, 80)
      .activation.tanh()
      .layer.dense(80, 80)
      .activation.tanh()
      .layer.dense(80, 80)
      .activation.tanh()
      .layer.dense(80, 1)
      .activation.tanh()
      .build();
    LossFunction lossFunction = new MeanSquareError();
    float learningRate = 0.00005f;
  
    private static final int MAX_STEPS = 800;
    private static final float CURVE_COUNT = 2;
  
    public Scene() {
      setBackground(Color.black);
      startAsyncThreads();
    }
    
    float createY(float y) {
      return (float) Math.sin(Math.toRadians(y * 360 * CURVE_COUNT));
    }
    
    void train() {
      ArrayList<Float> array = new ArrayList<>();
      for (int x = 0; x < MAX_STEPS; x++)
        array.add((float) x / MAX_STEPS);
      Collections.shuffle(array);
      
      for (float x : array) {
        float y = createY(x);
        neuralNetwork.trainSingle(
          lossFunction,
          new float[] { x },
          new float[] { y },
          learningRate
        );
      }
    }
    
    @Override
    public void paint(Graphics graphics) {
      super.paint(graphics);
  
      Graphics2D graphics2D = (Graphics2D) graphics;
      graphics2D.setStroke(new BasicStroke(2f));
      graphics2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
      
      int lastReal = -1;
      int lastFake = -1;
      for(int x = 0; x < this.getWidth(); x += 4) {
        float input = (float) x / this.getWidth();
        int yReal = (int) ((createY(input) / 8d + 0.5) * getHeight());
        float output = neuralNetwork.predictThreadSafe(input).data[0][0];
        int yFake = (int) ((output / 8d + 0.5) * getHeight());
  
        if(lastReal != -1) {
          graphics.setColor(Color.green);
          graphics.drawLine(x - 4, lastReal, x, yReal);
          graphics.setColor(Color.white);
          graphics.drawLine(x - 4, lastFake, x, yFake);
        }
        lastReal = yReal;
        lastFake = yFake;
      }
    }
    
    void startAsyncThreads() {
      train();
  
      new Thread(() -> {
        while (true) {
          this.repaint();
        }
      }).start();
      
      new Thread(() -> {
        try {
          Thread.sleep(1500);
        } catch (InterruptedException e) {
          throw new RuntimeException(e);
        }
        while (true) {
          train();
        }
      }).start();
    }
  }
}