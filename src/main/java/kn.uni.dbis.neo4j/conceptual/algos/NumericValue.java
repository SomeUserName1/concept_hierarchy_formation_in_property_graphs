package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.Locale;

/**
 * Holder for numeric values, aggregates from single values a gaussian/normal distribution.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class NumericValue extends Value {
  /**
   * Mean of the gaussian.
   */
  private float mean;
  /**
   * standard deviation of the gaussian.
   */
  private float std;
  /**
   * Used for Welford's and Chan's methods to compute the mean and variance incrementally and based on partitons.
   */
  private float m2;

  /**
   * Constructor for a single Number instance.
   * Initializes the number as mean with 0 stddev.
   *
   * @param nr value to set mean to.
   */
  public NumericValue(final Number nr) {
    this.setCount(1);
    this.mean = nr.floatValue();
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
  private NumericValue(final int count, final float mean, final float std, final float m2) {
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
      final float delta = v.mean - this.mean;
      final float mean = this.mean + delta * (float) v.getCount() / (float) totalCount;
      final float m2x = this.m2 + v.m2 + delta * delta * (this.getCount() * v.getCount()) / totalCount;
      this.mean = mean;
      this.std = (float) Math.sqrt(m2x / totalCount);
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
  float getStd() {
    return this.std;
  }

  /**
   * Getter method for the mean.
   *
   * @return std of the gaussian representing the NumericValue.
   */
  float getMean() {
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

  @Override
  public boolean equals(final Object o) {
    if (o instanceof NumericValue) {
      final NumericValue other = (NumericValue) o;
      return Float.compare(this.mean, other.mean) == 0 && Float.compare(this.std, other.std) == 0;
    } else {
      return false;
    }
  }

  /**
   * Returns a string that is formatted to be used in a latex tabular environment.
   * @return a sting representation of the node for letx tables
   */
  @Override
  public String toTexString() {
    String[] val = {String.format(Locale.US, "%.3f", this.mean), String.format(Locale.US, "%.3f", this.std)};
    return val[0].length() + val[1].length() < 30 ? "Numeric &  mean= " + val[0] + ", std=" + val[1] + " & "
        : "Numeric &  Value too large to display & ";
  }
}
