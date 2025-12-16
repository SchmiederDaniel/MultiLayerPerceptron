package neuralnetwork.activation;

import neuralnetwork.math.Tensor;
import static java.lang.Math.*;

public class GELU extends Activation {

    // Constant for the standard normal CDF
    private static final double SQRT_2_INV = 1.0 / sqrt(2.0);

    /**
     * High-quality approximation for the error function erf(x).
     * This is a standard approximation often used when Math.erf is unavailable.
     * Based on: C. Hastings, Jr., "Approximations for digital computers", 1955.
     *
     * @param z the input value
     * @return the value of the error function erf(z)
     */
    private double fastErf(double z) {
        // Coefficients for the approximation
        double t = 1.0 / (1.0 + 0.5 * abs(z));
        double t_sq = t * t;
        double poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
                t * (-1.453152027 + t * 1.061405429))));
        double result = 1.0 - poly * exp(-z * z);
        return signum(z) * result;
    }

    /**
     * Standard Gaussian Cumulative Distribution Function (CDF).
     * Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
     *
     * @param x the input value
     * @return the value of Phi(x)
     */
    private double phi(double x) {
        return 0.5 * (1.0 + fastErf(x * SQRT_2_INV));
    }

    /**
     * GELU activation function.
     * GELU(x) = x * Phi(x)
     *
     * @param x the input value
     * @return the activated value
     */
    private float gelu(float x) {
        return (float) (x * phi(x));
    }

    /**
     * Derivative of the GELU activation function.
     * GELU'(x) = Phi(x) + x * Phi'(x)
     * where Phi'(x) is the Standard Normal PDF: Phi'(x) = N(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
     *
     * @param x the input value
     * @return the derivative value
     */
    private float geluPrime(float x) {
        double x_double = x;
        double cdf = phi(x_double);
        
        // Standard Normal PDF (N(x) = Phi'(x))
        double pdf = (1.0 / sqrt(2.0 * PI)) * exp(-x_double * x_double / 2.0);
        
        // GELU'(x) = Phi(x) + x * N(x)
        return (float) (cdf + x_double * pdf);
    }

    @Override
    protected Tensor activation(Tensor input) {
        // Apply the GELU function element-wise
        return input.applyOperation(this::gelu);
    }

    @Override
    protected Tensor activationPrime(Tensor input) {
        // Apply the GELU derivative element-wise
        return input.applyOperation(this::geluPrime);
    }
}