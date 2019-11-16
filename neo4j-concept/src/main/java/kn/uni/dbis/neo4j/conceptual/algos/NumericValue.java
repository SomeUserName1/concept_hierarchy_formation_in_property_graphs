package kn.uni.dbis.neo4j.conceptual.algos;

import com.google.common.util.concurrent.AtomicDouble;

/**
 * Holder for numeric values, aggregates from single values a gaussian/normal distribution.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class NumericValue extends Value {
  /**
   * Mean of the gaussian.
   */
  private AtomicDouble mean;
  /**
   * standard deviation of the gaussian.
   */
  private AtomicDouble std;
  /**
   * Used for Welford's and Chan's methods to compute the mean and variance incrementally and based on partitons.
   */
  private AtomicDouble m2;

  /**
   * Constructor for a single Number instance.
   * Initializes the number as mean with 0 stddev.
   *
   * @param nr value to set mean to.
   */
  public NumericValue(final Number nr) {
    this.setCount(1);
    this.mean = new AtomicDouble(nr.doubleValue());
    this.std = new AtomicDouble(0.0f);
    this.m2 = new AtomicDouble(0.0f);
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
    this.mean = new AtomicDouble(mean);
    this.std = new AtomicDouble(std);
    this.m2 = new AtomicDouble(m2);
  }

  /**
   * returns the count of the updated node.
   * @param other node to incorporate
   */
  public void update(final Value other) {
    if (other instanceof NumericValue) {
      final NumericValue v = (NumericValue) other;
      final int totalCount = this.getCount() + v.getCount();
      final double delta = v.mean.get() - this.mean.get();
      final double mean = this.mean.get() + delta * (double) v.getCount() / (double) totalCount;
      final double m2x = this.m2.get() + v.m2.get() + delta * delta * (this.getCount() * v.getCount())
          / totalCount;
      this.mean.set(mean);
      this.std.set(Math.sqrt(m2x / totalCount));
      this.m2.set(m2x);
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
    return this.std.get();
  }

  /**
   * Getter method for the mean.
   *
   * @return std of the gaussian representing the NumericValue.
   */
  double getMean() {
    return this.mean.get();
  }

  @Override
  public Value copy() {
    return new NumericValue(this.getCount(), this.mean.get(), this.std.get(), this.m2.get());
  }

  @Override
  public String toString() {
    return "NumericValue:  count=" + this.getCount() + " mean= " + this.mean.get()
            + " std=" + this.std.get();
  }

  public String toTexString() {
    return "Numeric &  mean= " + this.mean.get() + ", std=" + this.std.get() + " &";
  }
}
