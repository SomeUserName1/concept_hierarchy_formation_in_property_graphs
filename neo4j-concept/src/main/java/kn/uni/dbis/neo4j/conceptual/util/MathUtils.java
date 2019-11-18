package kn.uni.dbis.neo4j.conceptual.util;

/**
 * Utilities for doing maths.
 *  @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public final class MathUtils {
  /**
   * Hidden default constructor.
   */
  private MathUtils() {
    // NOOP
  }

  /**
   * computes the logarithm to the basis 2 without numeric instability (as when dividing Math.log(x) / Match.log(2)).
   * @param bits the number to take the binary logarithm of
   * @return log_2(x) as integer rounded down
   */
  public static int log2(final int bits) {
    if (bits == 0) {
      return 0;
    }
    return 31 - Integer.numberOfLeadingZeros(bits);
  }
}
