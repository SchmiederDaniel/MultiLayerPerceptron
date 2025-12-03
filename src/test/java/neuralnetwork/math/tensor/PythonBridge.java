package neuralnetwork.math.tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

public class PythonBridge {
    /**
     * Example usage.
     */
    public static void main(String[] args) {
        try {
            PythonResult result = new PythonBuilder()
                .append("a = np.array([[1, -1, 3], [1, 1, 3]])")
                .append("b = np.array([3, 5, 6])")
                .append("c = a / b")
                .append("print(c)")
                .execute();
            String output = result.getOutput();
            System.out.println("Raw Output:\n" + output);
            Object parsed = result.parseToFloat();
            if (parsed instanceof Object[]) {
                Object[] parsedArray = (Object[]) parsed;
                System.out.println("Parsed Output: " + Arrays.deepToString(parsedArray));
            } else if (parsed instanceof float[]) {
                System.out.println("Parsed Output: " + Arrays.toString((float[]) parsed));
            } else {
                System.out.println("not parsable: " + parsed);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        try {
            PythonResult result = new PythonBuilder()
                .append("c = np.array([[1.0, np.inf], [-np.inf, np.nan]])")
                .append("print(c)")
                .execute();
            String output = result.getOutput();
            System.out.println("Raw Output:\n" + output);
            Object parsed = NumpyParser.parse(output);
            if (parsed instanceof Object[]) {
                Object[] parsedArray = (Object[]) parsed;
                System.out.println("Parsed Output: " + Arrays.deepToString(parsedArray));
            } else if (parsed instanceof float[]) {
                System.out.println("Parsed Output: " + Arrays.toString((float[]) parsed));
            } else {
                System.out.println("not parsable: " + parsed);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    /**
     * Builder class to construct Python scripts line by line.
     */
    public static class PythonBuilder {
        private StringBuilder scriptContent;
        private String tempFileName = "temp_script.py";
        private String pythonCommand = "python"; // Or "python3" depending on environment
        
        public PythonBuilder() {
            this.scriptContent = new StringBuilder();
            append("import numpy as np");
        }
        
        public PythonBuilder append(String line) {
            this.scriptContent.append(line).append("\n");
            return this;
        }
        
        public void setPythonCommand(String cmd) {
            this.pythonCommand = cmd;
        }
        
        /**
         * Writes the script to a file and executes it.
         */
        public PythonResult execute() throws IOException, InterruptedException {
            // 1. Write file
            File scriptFile = new File(tempFileName);
            try (FileWriter writer = new FileWriter(scriptFile)) {
                writer.write(scriptContent.toString());
            }
            
            // 2. Prepare Process
            ProcessBuilder pb = new ProcessBuilder(pythonCommand, tempFileName);
            pb.redirectErrorStream(true); // Merge stderr into stdout
            
            // 3. Start Process
            Process process = pb.start();
            
            // 4. Capture Output
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }
            
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new RuntimeException("Python script failed with exit code " + exitCode + "\nPython messagge:\n" + output);
            }
            
            // Cleanup
            if (scriptFile.exists()) {
                scriptFile.delete();
            }
            
            return new PythonResult(output.toString().trim(), exitCode);
        }
    }
    
    /**
     * Result object containing the raw output and parsing logic.
     */
    public static class PythonResult {
        private String rawOutput;
        private int exitCode;
        
        public PythonResult(String rawOutput, int exitCode) {
            this.rawOutput = rawOutput;
            this.exitCode = exitCode;
        }
        
        public String getOutput() {
            return rawOutput;
        }
        
        public int getExitCode() {
            return exitCode;
        }
        
        /**
         * Parses the rawOutput String to a java float array.
         * The dimension of the array can vary:
         * float[], float[][], float[][][] and float[][][][]
         * @return
         */
        public Object parseToFloat() {
            return NumpyParser.parse(rawOutput);
        }
    }
}