package de.darkandblue.neuralnetwork.models;

import de.darkandblue.neuralnetwork.NetworkBuilder;
import de.darkandblue.neuralnetwork.NeuralNetwork;
import de.darkandblue.neuralnetwork.initialization.Distribution;
import de.darkandblue.neuralnetwork.initialization.NormalDistribution;
import de.darkandblue.neuralnetwork.layer.Layer;
import de.darkandblue.neuralnetwork.lossfunction.AbsoluteLoss;
import de.darkandblue.neuralnetwork.lossfunction.LossFunction;
import de.darkandblue.neuralnetwork.math.NumpyArray;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

public class EmbeddingTest extends JPanel {
    private static final float learningRate = 0.001f;
    HashMap<String, Word> wordMap;

    NeuralNetwork network = new NetworkBuilder()
            .activation.tanh()
            .layer.dense(10, 30)
            .activation.tanh()
            .layer.dense(30, 30)
            .activation.tanh()
            .layer.dense(30, 30)
            .activation.tanh()
            .layer.dense(30, 10)
            .activation.tanh()
            .build();
    LossFunction loss = new AbsoluteLoss();

    public static void main(String[] args) {
        new EmbeddingTest();
    }

    EmbeddingTest() {
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(this);
        frame.setSize(100, 100);
        frame.setVisible(true);
        frame.setLocationRelativeTo(null);
        start();
    }

    void start() {
        System.out.println("Start loading Text");
        String text = loadText();
        System.out.println("Loaded Text");
        String[] converted = convertText(text);
        System.out.println("Converted Text " + converted.length);
        Set<String> reduced = reduce(converted);
        System.out.println("Reduced Text " + reduced.size());
        wordMap = extractNeighbors(reduced);
        train(converted, wordMap);
        predict(converted, wordMap);
    }

    void predict(String[] converted, HashMap<String, Word> wordMap) {
        new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                double error = 0;
                int count = 0;
                for (int i = 2; i < 500 - 2; i++) {
                    String wordText = converted[i];
                    Word word = wordMap.get(wordText);
                    if (word == null) {
                        System.out.println("No word found for " + wordText);
                        continue;
                    }

                    for (int j = -2; j < 2; j++) {
                        if (j != 0) {
                            String neighborText = converted[i + j];
                            Word neighbor = wordMap.get(neighborText);
                            if (neighbor == null) {
                                System.out.println("No neighbor found for " + neighborText);
                                continue;
                            }
                            float[] forward = word.forward(word.vector).flattenArray();
                            NumpyArray pred = network.predictThreadSafe(forward);
                            float err = loss.loss(neighbor.vector, pred);
                            error += err;
                            count++;
                        }
                    }
                }
                System.out.println("Loss: " + error / count);
                repaint();
            }
        }).start();
    }

    @Override
    public void paint(Graphics graphics) {
        super.paint(graphics);
        if (wordMap != null) {
            String[][] pairs = new String[][]{{"queen", "king"}, {"woman", "man"}};
            int i = 0;
            for (String[] pair : pairs) {
                String first = pair[0];
                String second = pair[1];
                Word firstWord = wordMap.get(first);
                Word secondWord = wordMap.get(second);
                NumpyArray firstForward = firstWord.forward(firstWord.vector);
                NumpyArray secondForward = secondWord.forward(secondWord.vector);
                NumpyArray diff = firstForward.subtract(secondForward);

                graphics.drawString(firstWord.text + ", " + secondWord.text
                        + ": vector: " + Arrays.toString(diff.flattenArray()), 10, 10 + i * 20);
                i++;
            }
        }
    }

    void train(String[] converted, HashMap<String, Word> wordMap) {
        new Thread(() -> {
            while (true) {
                for (int i = 2; i < converted.length - 2; i++) {
                    String wordText = converted[i];
                    Word word = wordMap.get(wordText);
                    if (word == null) {
                        System.out.println("No word found for " + wordText);
                        continue;
                    }

                    for (int j = -2; j < 2; j++) {
                        if (j != 0) {
                            String neighborText = converted[i + j];
                            Word neighbor = wordMap.get(neighborText);
                            if (neighbor == null) {
                                System.out.println("No neighbor found for " + neighborText);
                                continue;
                            }
                            float[] forward = word.forward(word.vector).flattenArray();
                            network.trainSingle(
                                    word,
                                    loss,
                                    forward,
                                    neighbor.vector.flattenArray(),
                                    learningRate);
                        }
                    }
                }
            }
        }).start();
    }

    HashMap<String, Word> extractNeighbors(Set<String> reduced) {
        HashMap<String, Word> wordMap = new HashMap<>();

        for (String wordText : reduced) {
            Word word = new Word(wordText, 10, new NormalDistribution());
            wordMap.put(wordText, word);
        }
        return wordMap;
    }

    String loadText() {
        try {
            byte[] bytes = this.getClass().getClassLoader().getResourceAsStream("MovieCorpus.txt").readAllBytes();
            String text = new String(bytes, StandardCharsets.UTF_8);
            return text;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    Set<String> reduce(String[] converted) {
        SortedSet<String> reduced = new TreeSet<>(Arrays.asList(converted));

        LinkedHashSet<String> sortedHashSet = reduced.stream()
                .sorted()
                .collect(Collectors.toCollection(LinkedHashSet::new));

        return sortedHashSet;
    }

    String[] convertText(String text) {
        text = text.toLowerCase();
        text = text.replaceAll("_", " ");
        text = text.replaceAll("'", "");
        text = text.replaceAll("´", "");
        text = text.replaceAll("\n", " ");
        text = text.replaceAll("\\d+", " ");
        text = text.replaceAll("[^\\w\\s]+", " ");
        text = text.replaceAll("\\s{2,}", " ");

        String[] result = text.split(" ");
        return result;
    }

    class Word extends Layer {
        public NumpyArray weights; // shape: (input_size, 1)
        public NumpyArray vector;
        public NumpyArray bias;    // shape: (input_size, 1)
        String text;
        NumpyArray input;

        public Word(String text, int input_size, Distribution distribution) {
            this.text = text;
            float[][] clone = new float[input_size][1];
            for (int i = 0; i < clone.length; i++) {
                clone[i][0] = 1;
            }
            weights = new NumpyArray(clone);

            clone = new float[input_size][1];
            for (int i = 0; i < clone.length; i++) {
                clone[i][0] = 0;
            }
            bias = new NumpyArray(clone);
            vector = new NumpyArray(input_size, 1);
            vector.randomize(-1, 1);
        }

        public Word(String text, NumpyArray weights, NumpyArray bias) {
            this.text = text;
            this.weights = weights;
            this.bias = bias;
        }

        @Override
        public NumpyArray forward(NumpyArray input) {
            this.input = input;
            // elementwise: output = input * weight + bias
            return input.multiply(weights).add(bias);
        }

        @Override
        public NumpyArray backward(NumpyArray output_gradient, float lr) {
            // output_gradient shape: (input_size, 1)

            // ∂L/∂w = input * output_gradient (elementwise)
            NumpyArray weights_gradient = input.multiply(output_gradient);

            // ∂L/∂b = output_gradient
            NumpyArray bias_gradient = output_gradient;

            // ∂L/∂input = w * output_gradient (elementwise)
            NumpyArray input_gradient = weights.multiply(output_gradient);

            // Update parameters
            weights = weights.subtract(weights_gradient.multiply(lr));
            bias = bias.subtract(bias_gradient.multiply(lr));

            return input_gradient;
        }

        @Override
        public Layer deepCopy() {
            return new Word(text, weights.copy(), bias.copy());
        }

        @Override
        public String toString() {
            return "ElementwiseDense " + weights.dimension() + " " + bias.dimension();
        }
    }
}