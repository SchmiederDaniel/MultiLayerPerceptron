package neuralnetwork.math.tensor;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NumpyParser {
    
    /**
     * Public API
     **/
    public static Object parse(String input) {
        // Safety check for empty input
        if (input == null || input.trim().isEmpty()) return new float[0];
        
        Tokenizer tokenizer = new Tokenizer(input);
        Node root = parseNode(tokenizer);
        
        int depth = getDepth(root);
        int[] shape = getShape(root, depth);
        
        return toPrimitiveArray(root, depth, shape, 0);
    }
    
    // ============================================================
    // Parsing AST
    // ============================================================
    
    private static Node parseNode(Tokenizer t) {
        t.skipWhitespace();
        
        // Array?
        if (t.peek() == '[') {
            t.consume('[');
            List<Node> list = new ArrayList<>();
            
            while (true) {
                t.skipWhitespace();
                
                if (t.peek() == ']') {
                    t.consume(']');
                    break;
                }
                
                list.add(parseNode(t));
                t.skipWhitespace();
                
                if (t.peek() == ',') {
                    t.consume(',');
                } else {
                    // NumPy sometimes uses whitespace instead of commas
                }
            }
            
            return new Node(list.toArray(new Node[0]));
        }
        
        // Number
        return new Node(t.readFloat());
    }
    
    // ============================================================
    // Shape inference
    // ============================================================
    
    private static int getDepth(Node n) {
        int d = 0;
        while (n.isArray()) {
            d++;
            if (n.array.length == 0) break; // handle empty arrays
            n = n.array[0];
        }
        return d;
    }
    
    private static int[] getShape(Node n, int depth) {
        int[] shape = new int[depth];
        Node current = n;
        
        for (int i = 0; i < depth; i++) {
            if (current.array == null || current.array.length == 0) {
                shape[i] = 0;
                break;
            }
            shape[i] = current.array.length;
            current = current.array[0];
        }
        
        return shape;
    }
    
    // ============================================================
    // Convert AST → primitive float array
    // ============================================================
    private static Object toPrimitiveArray(Node n, int depth, int[] shape, int level) {
        if (level == depth) {
            // leaf node: return primitive float
            return n.value;
        }
        
        int size = shape[level];
        
        // Determine the component type for this level
        Class<?> componentType = getArrayClass(shape, level + 1);
        Object arr = Array.newInstance(componentType, size);
        
        for (int i = 0; i < size; i++) {
            Object child = toPrimitiveArray(n.array[i], depth, shape, level + 1);
            
            if (child instanceof Float f) {
                Array.setFloat(arr, i, f);
            } else {
                Array.set(arr, i, child);
            }
        }
        
        return arr;
    }
    
    private static Class<?> getArrayClass(int[] shape, int level) {
        if (level >= shape.length) return float.class;
        
        Class<?> type = float.class;
        for (int i = level; i < shape.length; i++) {
            type = Array.newInstance(type, 0).getClass();
        }
        return type;
    }
    
    // ============================================================
    // AST Node
    // ============================================================
    
    private static class Node {
        Float value;      // if number
        Node[] array;     // if list
        
        Node(Float v) {
            this.value = v;
        }
        
        Node(Node[] a) {
            this.array = a;
        }
        
        boolean isArray() {
            return array != null;
        }
    }
    
    // ============================================================
    // Tokenizer (Updated for Edge Cases)
    // ============================================================
    
    private static class Tokenizer {
        private final String s;
        private int i = 0;
        
        Tokenizer(String s) {
            this.s = s;
        }
        
        char peek() {
            skipWhitespace();
            if (i >= s.length()) return '\0';
            return s.charAt(i);
        }
        
        void consume(char c) {
            skipWhitespace();
            if (peek() != c)
                throw new RuntimeException("Expected '" + c + "' at position " + i + " found '" + peek() + "'");
            i++;
        }
        
        void skipWhitespace() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i)))
                i++;
        }
        
        float readFloat() {
            skipWhitespace();
            int start = i;
            boolean isNegative = false;
            
            // 1. Consume Sign
            if (i < s.length() && (s.charAt(i) == '+' || s.charAt(i) == '-')) {
                if (s.charAt(i) == '-') isNegative = true;
                i++;
            }
            
            // 2. Check for Special Values (nan, inf)
            // NumPy outputs "nan", "inf", "-inf"
            if (i < s.length() && Character.isLetter(s.charAt(i))) {
                int wordStart = i;
                while (i < s.length() && Character.isLetter(s.charAt(i))) {
                    i++;
                }
                String word = s.substring(wordStart, i).toLowerCase();
                
                if (word.equals("nan")) {
                    return Float.NaN;
                } else if (word.contains("inf")) {
                    // matches "inf" or "infinity"
                    return isNegative ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
                } else {
                    throw new RuntimeException("Unknown token: " + word);
                }
            }
            
            // 3. Normal Digits and decimal
            while (i < s.length() && (Character.isDigit(s.charAt(i)) || s.charAt(i) == '.'))
                i++;
            
            // 4. Exponent (e.g., 1.23e-5)
            if (i < s.length() && (s.charAt(i) == 'e' || s.charAt(i) == 'E')) {
                i++;
                if (i < s.length() && (s.charAt(i) == '+' || s.charAt(i) == '-'))
                    i++;
                while (i < s.length() && Character.isDigit(s.charAt(i)))
                    i++;
            }
            
            String token = s.substring(start, i);
            try {
                return Float.parseFloat(token);
            } catch (NumberFormatException e) {
                // Fallback or better error message
                throw new RuntimeException("Failed to parse number: " + token, e);
            }
        }
    }
    
    // Helpers for tolerant comparisons and parsing from Python
    private static void assertFloatArrayEquals(float[] expected, float[] actual, float eps) {
        assertEquals(expected.length, actual.length, "Length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], eps, "Mismatch at index " + i);
        }
    }
    
    private static void assertFloat2DArrayEquals(float[][] expected, float[][] actual, float eps) {
        assertEquals(expected.length, actual.length, "Row count mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertFloatArrayEquals(expected[i], actual[i], eps);
        }
    }
    
    public static float[] to1D(Object parsed) {
        if (parsed instanceof float[] fa) return fa;
        if (parsed instanceof Object[] oa && oa.length > 0 && oa[0] instanceof Float) {
            float[] out = new float[oa.length];
            for (int i = 0; i < oa.length; i++) out[i] = ((Float) oa[i]);
            return out;
        }
        throw new AssertionError("Unexpected parsed 1D type: " + (parsed == null ? "null" : parsed.getClass()));
    }
    
    public static float[][] to2D(Object parsed) {
        if (parsed instanceof float[][] faa) return faa;
        if (parsed instanceof Object[] oa && oa.length > 0) {
            if (oa[0] instanceof float[] row) {
                float[][] out = new float[oa.length][];
                for (int i = 0; i < oa.length; i++) out[i] = (float[]) oa[i];
                return out;
            }
            if (oa[0] instanceof Object[] ob && ob.length > 0 && ob[0] instanceof Float) {
                float[][] out = new float[oa.length][];
                for (int i = 0; i < oa.length; i++) {
                    Object[] rowObj = (Object[]) oa[i];
                    float[] rowArr = new float[rowObj.length];
                    for (int j = 0; j < rowObj.length; j++) rowArr[j] = (Float) rowObj[j];
                    out[i] = rowArr;
                }
                return out;
            }
        }
        throw new AssertionError("Unexpected parsed 2D type: " + (parsed == null ? "null" : parsed.getClass()));
    }
}