package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.Objects;

/**
 * Holder for nominal values, representing Strings, characters, booleans and IDs as String.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class NominalValue extends Value {
  /**
   * the nominal value as string.
   */
  private final String str;

  /**
   * Constructor using a string.
   *
   * @param value the value to store
   */
  public NominalValue(final String value) {
    this.setCount(1);
    this.str = value;
  }

  /**
   * copy Constructor.
   *
   * @param value the value to store
   * @param count count to be set
   */
  private NominalValue(final int count, final String value) {
    this.setCount(count);
    this.str = value;
  }

  /**
   * Constructor using a boolean.
   *
   * @param value the value to store
   */
  NominalValue(final boolean value) {
    this.setCount(1);
    this.str = value ? "true" : "false";
  }

  /**
   * Constructor using a character.
   *
   * @param value the value to store
   */
  NominalValue(final char value) {
    this.setCount(1);
    this.str = Character.toString(value);
  }

  @Override
  public Value copy() {
    return new NominalValue(this.getCount(), this.str);
  }

  @Override
  public boolean equals(final Object o) {
    if (o instanceof NominalValue)  {
      final NominalValue n = (NominalValue) o;
      return n.str.equals(this.str);
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.str);
  }

  @Override
  public void update(final Value other) {
    if (other instanceof NominalValue) {
      final NominalValue n = (NominalValue) other;
      if (n.str.equals(this.str)) {
        this.setCount(this.getCount() + n.getCount());
      }
    } else {
      throw new RuntimeException("updated with wrong type!");
    }
  }

  @Override
  public String toString() {
    return "NominalValue count=" + this.getCount() + " string=" + this.str;
  }
}
