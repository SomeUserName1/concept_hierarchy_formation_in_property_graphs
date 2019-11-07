package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.Objects;

/**
 * Holder for nominal values, representing Strings, characters, booleans and IDs as String.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class NominalValue implements Value, Cloneable {
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
    this.str = value;
  }

  /**
   * Constructor using a boolean.
   *
   * @param value the value to store
   */
  NominalValue(final boolean value) {
    this.str = value ? "true" : "false";
  }

  /**
   * Constructor using a character.
   *
   * @param value the value to store
   */
  NominalValue(final char value) {
    this.str = Character.toString(value);
  }

  /**
   * Constructor using a long.
   *
   * @param value the value to store
   */
  NominalValue(final long value) {
    this.str = Long.toHexString(value);
  }

  @Override
  public Value clone() {
    try {
      super.clone();
    } catch (final CloneNotSupportedException e) {
      e.printStackTrace();
    }
    return new NominalValue(this.str);
  }

  @Override
  public boolean equals(final Object o) {
    if (o instanceof NominalValue)  {
      final NominalValue n = (NominalValue) o;
      return n.str.equals(this.str);
    } else if (o instanceof String) {
      final String str = (String) o;
      return str.equals(this.str);
    } else if (o instanceof Boolean) {
      final String bool = (Boolean) o ? "true" : "false";
      return bool.equals(this.str);
    } else if (o instanceof Character) {
      final String c = Character.toString((Character) o);
      return c.equals(this.str);
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(this.str);
  }
}
