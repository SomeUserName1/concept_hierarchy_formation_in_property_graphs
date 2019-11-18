package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.Objects;

/**
 * Holder for numeric values, aggregates from single values a gaussian/normal distribution.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class NumericValue extends Value {
  /**
   * Mean of the gaussian.
   */
  private double mean;
  /**
   * standard deviation of the gaussian.
   */
  private double std;
  /**
   * Used for Welford's and Chan's methods to compute the mean and variance incrementally and based on partitons.
   */
  private double m2;

  /**
   * Constructor for a single Number instance.
   * Initializes the number as mean with 0 stddev.
   *
   * @param nr value to set mean to.
   */
  public NumericValue(final Number nr) {
    this.setCount(1);
    this.mean = nr.doubleValue();
    this.std = 0.0f;
    this.m2 = 0.0f;
  }

  /**
   * Copy Constructor: takes mean and std to construct a NumericValue with the given args.
   *
   * @param count count to be set
   * @param mean mean of the gaussian to be initialized.
   * @param std  std of the gaussian to be initialized.
   * @param m2 used by welford and chans online algos for mean and variance
   */
  private NumericValue(final int count, final double mean, final double std, final double m2) {
    this.setCount(count);
    this.mean = mean;
    this.std = std;
    this.m2 = m2;
  }

  /**
   * returns the count of the updated node.
   * @param other node to incorporate
   */
  public void update(final Value other) {
    if (other instanceof NumericValue) {
      final NumericValue v = (NumericValue) other;
      final int totalCount = this.getCount() + v.getCount();
      final double delta = v.mean - this.mean;
      final double mean = this.mean + delta * (double) v.getCount() / (double) totalCount;
      final double m2x = this.m2 + v.m2 + delta * delta * (this.getCount() * v.getCount()) / totalCount;
      this.mean = mean;
      this.std = Math.sqrt(m2x / totalCount);
      this.m2 = m2x;
      this.setCount(totalCount);
    } else {
      throw new RuntimeException("updated with wrong type!");
    }
  }

  /**
   * Getter method for the standard deviation.
   *
   * @return std of the gaussian representing the NumericValue.
   */
  double getStd() {
    return this.std;
  }

  /**
   * Getter method for the mean.
   *
   * @return std of the gaussian representing the NumericValue.
   */
  double getMean() {
    return this.mean;
  }

  @Override
  public Value copy() {
    return new NumericValue(this.getCount(), this.mean, this.std, this.m2);
  }

  @Override
  public String toString() {
    return "NumericValue:  count=" + this.getCount() + " mean= " + this.mean
            + " std=" + this.std;
  }
}
