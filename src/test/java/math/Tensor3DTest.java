package math;

import neuralnetwork.math.Matrix;
import neuralnetwork.math.Scalar;
import neuralnetwork.math.Tensor3D;
import neuralnetwork.math.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Tensor3DTest {

    private static final String A3_PY = "a = np.array([[[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]], [[-1.0, 0.0, 1.0],[2.0, -2.0, 3.0]]])";

    @Test
    void add_sub_divide_broadcast() throws Exception {
        Tensor3D A = new Tensor3D(new float[][][] {
            { { 1f, 2f, 3f }, { 4f, 5f, 6f } },
            { { -1f, 0f, 1f }, { 2f, -2f, 3f } }
        });

        // + Scalar
        Scalar s = new Scalar(2.5f);
        PythonResult rAddS = new PythonBuilder()
            .append(A3_PY)
            .append("c = a + 2.5")
            .execute("c");
        assertArrayEquals((Object[]) rAddS.parseToFloat(), ((Tensor3D) A.add(s)).getValues());

        // + Vector (broadcast along last dim)
        Vector v = new Vector(new float[] { 2f, -3f, 4f });
        PythonResult rAddV = new PythonBuilder()
            .append(A3_PY)
            .append("v = np.array([2.0, -3.0, 4.0])")
            .append("c = a + v")
            .execute("c");
        assertArrayEquals((Object[]) rAddV.parseToFloat(), ((Tensor3D) A.add(v)).getValues());

        // + Matrix (broadcast across first dim)
        Matrix mSame = new Matrix(new float[][] { { 1f, -1f, 0f }, { 0.5f, 0.5f, -0.5f } });
        PythonResult rAddM = new PythonBuilder()
            .append(A3_PY)
            .append("m = np.array([[1.0, -1.0, 0.0],[0.5, 0.5, -0.5]])")
            .append("c = a + m")
            .execute("c");
        assertArrayEquals((Object[]) rAddM.parseToFloat(), ((Tensor3D) A.add(mSame)).getValues());

        // element-wise matmul with same-shape 3D
        Tensor3D B = new Tensor3D(new float[][][] {
            { { 2f, -1f, 0.5f }, { 3f, 0.5f, -2f } },
            { { -1f, 2f, 3f }, { 0.5f, 1.5f, -0.5f } }
        });
        PythonResult rHad = new PythonBuilder()
            .append(A3_PY)
            .append("b = np.array([[[2.0,-1.0,0.5],[3.0,0.5,-2.0]],[[-1.0,2.0,3.0],[0.5,1.5,-0.5]]])")
            .append("c = a * b")
            .execute("c");
        assertArrayEquals((Object[]) rHad.parseToFloat(), ((Tensor3D) A.multiply(B)).getValues());

        // divide by vector (broadcast)
        PythonResult rDivV = new PythonBuilder()
            .append(A3_PY)
            .append("v = np.array([2.0, -3.0, 4.0])")
            .append("c = a / v")
            .execute("c");
        assertArrayEquals((Object[]) rDivV.parseToFloat(), ((Tensor3D) A.divide(v)).getValues());
    }

    @Test
    void mul_linear_algebra_batched() throws Exception {
        Tensor3D A = new Tensor3D(new float[][][] {
            { { 1f, 2f, 3f }, { 4f, 5f, 6f } },
            { { -1f, 0f, 1f }, { 2f, -2f, 3f } }
        });

        // A (2,2,3) * v(3) => (2,2) Matrix using matvec per batch
        Vector v = new Vector(new float[] { 2f, -3f, 4f });
        PythonResult r1 = new PythonBuilder()
            .append(A3_PY)
            .append("v = np.array([2.0, -3.0, 4.0])")
            .append("c = a @ v")
            .execute("c");
        assertArrayEquals((Object[]) r1.parseToFloat(), ((Matrix) A.matmul(v)).getValues());

        // A (2,2,3) * M(3,2) => (2,2,2) Tensor3D using matmul per batch
        Matrix m = new Matrix(new float[][] { { 2f, 0f }, { 1f, 3f }, { -1f, 2f } });
        PythonResult r2 = new PythonBuilder()
            .append(A3_PY)
            .append("M = np.array([[2.0, 0.0],[1.0, 3.0],[-1.0, 2.0]])")
            .append("c = a @ M")
            .execute("c");
        assertArrayEquals((Object[]) r2.parseToFloat(), ((Tensor3D) A.matmul(m)).getValues());

        // A (2,2,3) * B(2,3,2) => (2,2,2) batched matmul
        Tensor3D B = new Tensor3D(new float[][][] {
            { { 1f, 0f }, { 0f, 1f }, { 1f, 1f } },
            { { -1f, 2f }, { 0f, 1f }, { 2f, -1f } }
        });
        PythonResult r3 = new PythonBuilder()
            .append(A3_PY)
            .append("B = np.array([[[1.0,0.0],[0.0,1.0],[1.0,1.0]],[[-1.0,2.0],[0.0,1.0],[2.0,-1.0]]])")
            .append("c = np.matmul(a, B)")
            .execute("c");
        assertArrayEquals((Object[]) r3.parseToFloat(), ((Tensor3D) A.matmul(B)).getValues());
    }

    @Test
    void transpose_last_two_dims() throws Exception {
        Tensor3D A = new Tensor3D(new float[][][] {
            { { 1f, 2f, 3f }, { 4f, 5f, 6f } },
            { { -1f, 0f, 1f }, { 2f, -2f, 3f } }
        });
        PythonResult r = new PythonBuilder()
            .append(A3_PY)
            .append("c = np.transpose(a, (0,2,1))")
            .execute("c");
        assertArrayEquals((Object[]) r.parseToFloat(), ((Tensor3D) A.transpose()).getValues());
    }
}