package models.dataset;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;

public class MNISTLoader {
    public static List<int[]> readTrainImagesSafe() {
        return readImagesSafe("train-images.idx3-ubyte");
    }
    
    public static List<int[]> readTestImagesSafe() {
        return readImagesSafe("t10k-images.idx3-ubyte");
    }
    
    public static List<Integer> readTrainLabelsSafe() {
        return readTestLabelsSafe("train-labels.idx1-ubyte");
    }
    
    public static List<Integer> readTestLabelsSafe() {
        return readTestLabelsSafe("t10k-labels.idx1-ubyte");
    }
    
    private static List<int[]> readImagesSafe(String fileName) {
        byte[] raw = readResource(fileName);
        List<int[]> images = new ArrayList<>();
        byte[] imageBytes = new byte[raw.length - 16];
        System.arraycopy(raw, 16, imageBytes, 0, imageBytes.length);
        
        while (images.size() * 28 * 28 + 28 * 28 - 1 < imageBytes.length) {
            int[] image = new int[28 * 28];
            for (int pixelIndex = 0; pixelIndex < image.length; pixelIndex++) {
                int unsigned = imageBytes[images.size() * 28 * 28 + pixelIndex] & 0xff;
                image[pixelIndex] = unsigned;
            }
            images.add(image);
        }
        return images;
    }
    
    private static List<Integer> readTestLabelsSafe(String fileName) {
        byte[] raw = readResource(fileName);
        List<Integer> labels = new CopyOnWriteArrayList<>();
        byte[] labelBytes = new byte[raw.length - 8];
        System.arraycopy(raw, 8, labelBytes, 0, labelBytes.length);
        
        for (int i = 0; i < labelBytes.length; i++) {
            int unsigned = labelBytes[i] & 0xff;
            labels.add(unsigned);
        }
        
        return labels;
    }
    
    private static byte[] readResource(String name) {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(
            Objects.requireNonNull(MNISTLoader.class.getClassLoader().getResourceAsStream(name))))) {
            return in.readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}