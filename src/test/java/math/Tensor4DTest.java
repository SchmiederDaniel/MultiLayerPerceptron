package math;

import neuralnetwork.math.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Tensor4DTest {

    // Shape: (2,2,2,3)
    private static final String A4_PY = "a = np.array([[[[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]],[[7.0, 8.0, 9.0],[10.0, 11.0, 12.0]]], [[[ -1.0, 0.0, 1.0],[2.0, -2.0, 3.0]], [[-3.0, 4.0, -5.0],[6.0, -7.0, 8.0]]]])";

    private static Tensor4D buildA() {
        return new Tensor4D(new float[][][][]{
            {
                { { 1f, 2f, 3f }, { 4f, 5f, 6f } },
                { { 7f, 8f, 9f }, { 10f, 11f, 12f } }
            },
            {
                { { -1f, 0f, 1f }, { 2f, -2f, 3f } },
                { { -3f, 4f, -5f }, { 6f, -7f, 8f } }
            }
        });
    }

    @Test
    void add_sub_divide_broadcast_and_hadamard() throws Exception {
        Tensor4D A = buildA();

        // + Scalar
        PythonResult rAddS = new PythonBuilder()
            .append(A4_PY)
            .append("c = a + 2.5")
            .execute("c");
        assertArrayEquals((Object[]) rAddS.parseToFloat(), ((Tensor4D) A.add(new Scalar(2.5f))).getValues());

        // + Vector (broadcast along last dim)
        Vector v = new Vector(new float[] { 2f, -3f, 4f });
        PythonResult rAddV = new PythonBuilder()
            .append(A4_PY)
            .append("v = np.array([2.0, -3.0, 4.0])")
            .append("c = a + v")
            .execute("c");
        assertArrayEquals((Object[]) rAddV.parseToFloat(), ((Tensor4D) A.add(v)).getValues());

        // + Matrix (broadcast along first two dims)
        Matrix M = new Matrix(new float[][] { { 1f, -1f, 0.5f }, { -0.5f, 0.5f, -1.5f } });
        PythonResult rAddM = new PythonBuilder()
            .append(A4_PY)
            .append("M = np.array([[1.0, -1.0, 0.5],[-0.5, 0.5, -1.5]])")
            .append("c = a + M")
            .execute("c");
        assertArrayEquals((Object[]) rAddM.parseToFloat(), ((Tensor4D) A.add(M)).getValues());

        // + Tensor3D broadcast across first dimension (batch)
        Tensor3D T3 = new Tensor3D(new float[][][] {
            { { 1f, 0f, -1f }, { 0.5f, -0.5f, 1.5f } },
            { { -2f, 2f, 0f }, { 1f, -1f, 0f } }
        }); // shape (2,2,3) aligns with [d2,r,c]
        PythonResult rAddT3 = new PythonBuilder()
            .append(A4_PY)
            .append("T3 = np.array([[[1.0,0.0,-1.0],[0.5,-0.5,1.5]],[[-2.0,2.0,0.0],[1.0,-1.0,0.0]]])")
            .append("c = a + T3")
            .execute("c");
        assertArrayEquals((Object[]) rAddT3.parseToFloat(), ((Tensor4D) A.add(T3)).getValues());

        // Hadamard (element-wise)
        Tensor4D B4 = buildA();
        PythonResult rHad = new PythonBuilder()
            .append(A4_PY)
            .append("b = " + A4_PY.substring("a = ".length()))
            .append("c = a * b")
            .execute("c");
        assertArrayEquals((Object[]) rHad.parseToFloat(), ((Tensor4D) A.multiply(B4)).getValues());

        // Divide by vector broadcast
        PythonResult rDiv = new PythonBuilder()
            .append(A4_PY)
            .append("v = np.array([2.0, -3.0, 4.0])")
            .append("c = a / v")
            .execute("c");
        assertArrayEquals((Object[]) rDiv.parseToFloat(), ((Tensor4D) A.divide(v)).getValues());
    }

    @Test
    void mul_linear_algebra_batched() throws Exception {
        Tensor4D A = buildA();

        // A (2,2,2,3) @ v(3) -> (2,2,2) Tensor3D
        Vector v = new Vector(new float[] { 2f, -3f, 4f });
        PythonResult r1 = new PythonBuilder()
            .append(A4_PY)
            .append("v = np.array([2.0, -3.0, 4.0])")
            .append("c = np.matmul(a, v)")
            .execute("c");
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Tensor3D) A.matmul(v)).getValues());

        // A (2,2,2,3) @ M(3,2) -> (2,2,2,2)
        Matrix M = new Matrix(new float[][] { { 2f, 0f }, { 1f, 3f }, { -1f, 2f } });
        PythonResult r2 = new PythonBuilder()
            .append(A4_PY)
            .append("M = np.array([[2.0, 0.0],[1.0, 3.0],[-1.0, 2.0]])")
            .append("c = np.matmul(a, M)")
            .execute("c");
        assertArrayEquals((Object[]) r2.parseToFloat(), ((Tensor4D) A.matmul(M)).getValues());

        // A (2,2,2,3) @ B(2,2,3,2) -> (2,2,2,2)
        Tensor4D B4 = new Tensor4D(new float[][][][] {
            {
                { { 1f, 0f }, { 0f, 1f }, { 1f, 1f } },
                { { -1f, 2f }, { 0f, 1f }, { 2f, -1f } }
            },
            {
                { { 0f, 1f }, { 1f, 0f }, { -1f, 1f } },
                { { 2f, -2f }, { 1f, 1f }, { 0f, 1f } }
            }
        });
        PythonResult r3 = new PythonBuilder()
            .append(A4_PY)
            .append("B = np.array([[[[1.0,0.0],[0.0,1.0],[1.0,1.0]], [[-1.0,2.0],[0.0,1.0],[2.0,-1.0]]], [[[0.0,1.0],[1.0,0.0],[-1.0,1.0]], [[2.0,-2.0],[1.0,1.0],[0.0,1.0]]]])")
            .append("c = np.matmul(a, B)")
            .execute("c");
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Tensor4D) A.matmul(B4)).getValues());
    }

    @Test
    void transpose_last_two_dims() throws Exception {
        Tensor4D A = buildA();
        PythonResult r = new PythonBuilder()
            .append(A4_PY)
            .append("c = np.transpose(a, (0,1,3,2))")
            .execute("c");
        assertArrayEquals((Object[]) r.parseToFloat(), ((Tensor4D) A.transpose()).getValues());
    }
}