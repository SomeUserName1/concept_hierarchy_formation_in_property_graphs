package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.Objects;

/**
 * Holder for numeric values, aggregates from single values a gaussian/normal distribution.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class NumericValue implements Value, Cloneable {
  /**
   * Mean of the gaussian.
   */
  private double mean;
  /**
   * standard deviation of the gaussian.
   */
  private double std;

  /**
   * Constructor for a single Number instance.
   * Initializes the number as mean with 0 stddev.
   *
   * @param nr value to set mean to.
   */
  NumericValue(final Number nr) {
    this.mean = nr.doubleValue();
    this.std = 0.0f;
  }

  /**
   * Semi Copy Constructor: takes mean and std to construct a NumericValue with the given args.
   *
   * @param mean mean of the gaussian to be initialized.
   * @param std  std of the gaussian to be initialized.
   */
  private NumericValue(final double mean, final double std) {
    this.mean = mean;
    this.std = std;
  }

  /**
   * Getter method for the mean.
   *
   * @return mean of the gaussian representing the NumericValue.
   */
  public double getMean() {
    return this.mean;
  }

  /**
   * Setter for the mean.
   *
   * @param mean value to be set.
   */
  public void setMean(final double mean) {
    this.mean = mean;
  }

  /**
   * Getter method for the standard deviation.
   *
   * @return std of the gaussian representing the NumericValue.
   */
  public double getStd() {
    return this.std;
  }

  /**
   * Setter for the std.
   *
   * @param std value to be set.
   */
  public void setStd(final double std) {
    this.std = std;
  }

  @Override
  public Value clone() {
    try {
      super.clone();
    } catch (final CloneNotSupportedException e) {
      e.printStackTrace();
    }
    return new NumericValue(this.mean, this.std);
  }

  @Override
  public boolean equals(final Object o) {
    if (o instanceof NumericValue) {
      final NumericValue nr = (NumericValue) o;
      final double minDifference = this.mean - nr.getMean() > 0 ? this.mean - 3 * this.std - nr.mean + 3 * nr.std
          : nr.mean - 3 * nr.std - this.mean + 3 * this.std;
      return minDifference <= 0;
    } else if (o instanceof Number) {
      final double nr = ((Number) o).doubleValue();
      return nr > this.mean ? nr - this.mean + 3 * this.std <= 0 : this.mean - 3 * this.std - nr <= 0;
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.mean, this.std);
  }
}
