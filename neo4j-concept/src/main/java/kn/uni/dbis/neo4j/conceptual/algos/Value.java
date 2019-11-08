package kn.uni.dbis.neo4j.conceptual.algos;

/**
 * Interface for a Value to cluster using Cobweb.
 * I.e. it's held by the attributes map as value in the @link{ConceptNode} class.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public abstract class Value {
  /** counter for the occurrence of the value. */
  private int count;

  /**
   * given a primitive datatype or a String, decide which Value type to instantiate.
   *
   * @param o the object to cast. Should be a String, a Boolean, a char, an instance of the Number interface
   *          or a ConceptNode.
   * @return an appropriate Value Instance
   */
  static Value cast(final Object o) {
    if (o instanceof String) {
      return new NominalValue((String) o);
    } else if (o instanceof Boolean) {
      return new NominalValue((boolean) o);
    } else if (o instanceof Character) {
      return new NominalValue((Character) o);
    } else if (o instanceof Number) {
      return new NumericValue((Number) o);
    } else if (o instanceof ConceptNode) {
      return new ConceptValue((ConceptNode) o);
    } else {
      System.out.println("Encountered Property with unsupported type!");
      return new NominalValue("Unsupported Type!");
    }
  }

  /**
   * Used when computing the CU for the 4 operations of cobweb (insert, split, merge, recurse).
   *
   * @return returns a deep copy of the Value it's called on.
   */
  public abstract Value copy();

  /**
   * unpacks the held value and checks for equality depending on the ValueType (e.g. NominalValue checks
   * for String equivalence, NumericValue for overlap in the 3sigma space arround the means.
   *
   * @param o object to compare
   * @return a boolean inidcating equivalence
   */
  @Override
  public abstract boolean equals(Object o);

  @Override
  public abstract int hashCode();

  /**
   * Given another Value of the same type update the value of the current one.
   * @param other the Value to incorporate
   */
  public abstract void update(Value other);

  /**
   * get the count of the value.
   * @return the count of the value
   */
  public int getCount() {
    return this.count;
  }

  /**
   * Sets the count of a value.
   * @param count count to be set
   */
  void setCount(final int count) {
    this.count = count;
  }
}
