package kn.uni.dbis.neo4j.conceptual.algos;


import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Holder for nominal values, representing Strings, characters, booleans and IDs as String.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class NominalValue extends Value {
  /**
   * the nominal value as string.
   */
  private final AtomicReference<String> str;

  /**
   * Constructor using a string.
   *
   * @param value the value to store
   */
  public NominalValue(final String value) {
    this.setCount(1);
    this.str = new AtomicReference<>(value);
  }

  /**
   * copy Constructor.
   *
   * @param value the value to store
   * @param count count to be set
   */
  private NominalValue(final int count, final String value) {
    this.setCount(count);
    this.str = new AtomicReference<>(value);
  }

  /**
   * Constructor using a boolean.
   *
   * @param value the value to store
   */
  NominalValue(final boolean value) {
    this.setCount(1);
    this.str = value ? new AtomicReference<>("true") : new AtomicReference<>("false");
  }

  /**
   * Constructor using a character.
   *
   * @param value the value to store
   */
  NominalValue(final char value) {
    this.setCount(1);
    this.str = new AtomicReference<>(Character.toString(value));
  }

  @Override
  public Value copy() {
    return new NominalValue(this.getCount(), this.str.get());
  }

  @Override
  public boolean equals(final Object o) {
    if (o instanceof NominalValue)  {
      final NominalValue n = (NominalValue) o;
      return n.str.get().equals(this.str.get());
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.str.get());
  }

  @Override
  public void update(final Value other) {
    if (other instanceof NominalValue) {
      final NominalValue n = (NominalValue) other;
      if (n.str.get().equals(this.str.get())) {
        this.setCount(this.getCount() + n.getCount());
      }
    } else {
      throw new RuntimeException("updated with wrong type!");
    }
  }

  @Override
  public String toString() {
    return "NominalValue count=" + this.getCount() + " string=" + this.str.get();
  }

  String toTexString() {
    return "Nominal & " + this.str.get() + "& ";
  }
}
