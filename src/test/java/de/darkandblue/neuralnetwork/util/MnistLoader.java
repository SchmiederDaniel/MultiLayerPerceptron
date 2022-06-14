package de.darkandblue.neuralnetwork.util;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class MnistLoader {
  public static List<Integer> readLabels() throws IOException {
    List<Integer> labels = new CopyOnWriteArrayList<>();
    byte[] rawLabelBytes = readFileFromRessources("train-labels.idx1-ubyte");
    byte[] labelBytes = new byte[rawLabelBytes.length - 8];
    System.arraycopy(rawLabelBytes, 8, labelBytes, 0, labelBytes.length);
    
    for (int currentImage = 0; currentImage < 60000; currentImage++) {
      int unsigned = labelBytes[currentImage] & 0xff;
      labels.add(unsigned);
    }
    
    return labels;
  }
  
  public static List<int[]> readImages() throws IOException {
    List<int[]> images = new ArrayList<>();
    byte[] rawImageBytes = readFileFromRessources("train-images.idx3-ubyte");
    byte[] imageBytes = new byte[rawImageBytes.length - 16];
    System.arraycopy(rawImageBytes, 16, imageBytes, 0, imageBytes.length);
    
    for (int currentImage = 0; currentImage < 60000; currentImage++) {
      int[] image = new int[28 * 28];
      for (int pixelIndex = 0; pixelIndex < image.length; pixelIndex++) {
        int unsigned = imageBytes[currentImage * 28 * 28 + pixelIndex] & 0xff;
        image[pixelIndex] = unsigned;
      }
      images.add(image);
    }
    return images;
  }
  
  public static byte[] readFileFromRessources(String fileName) throws IOException {
    DataInputStream imageInputStream = new DataInputStream(
      new BufferedInputStream(MnistLoader.class.getClassLoader().getResourceAsStream(fileName)));
    byte[] rawBytes = imageInputStream.readAllBytes();
    imageInputStream.close();
    return rawBytes;
  }
}