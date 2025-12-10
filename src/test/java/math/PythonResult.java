package math;

/**
 * Result object containing the raw output and parsing logic.
 */
public class PythonResult {
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
     *
     * @return
     */
    public Object parseToFloat() {
        return NumpyParser.parse(rawOutput);
    }
}