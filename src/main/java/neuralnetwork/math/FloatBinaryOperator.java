package neuralnetwork.math;

/**
 * Represents an operation upon two {@code int}-valued operands and producing an
 * {@code int}-valued result.   This is the primitive type specialization of
 * {@link FloatBinaryOperator} for {@code float}.
 *
 * <p>This is a <a href="package-summary.html">functional interface</a>
 * whose functional method is {@link #applyAsFloat(float, float)}.

 * @since 1.8
 */
@FunctionalInterface
public interface FloatBinaryOperator {
    
    /**
     * Applies this operator to the given operands.
     *
     * @param left the first operand
     * @param right the second operand
     * @return the operator result
     */
    float applyAsFloat(float left, float right);
}