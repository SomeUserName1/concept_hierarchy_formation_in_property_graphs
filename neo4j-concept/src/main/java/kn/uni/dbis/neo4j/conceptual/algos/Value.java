package kn.uni.dbis.neo4j.conceptual.algos;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import javax.annotation.concurrent.ThreadSafe;

/**
 * Interface for a Value to cluster using Cobweb.
 * I.e. it's held by the attributes map as value in the @link{ConceptNode} class.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
@ThreadSafe
public abstract class Value {
  /** counter for the occurrence of the value. */
  private AtomicInteger count = new AtomicInteger();
  /** Lock. */
  private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

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
    }  else {
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
   * Given another Value of the same type update the value of the current one.
   * @param other the Value to incorporate
   */
  public abstract void update(Value other);

  /**
   * get the count of the value.
   * @return the count of the value
   */
  public int getCount() {
      return this.count.get();
  }

  /**
   * Sets the count of a value.
   * @param count count to be set
   */
  void setCount(final int count) {
    this.count.set(count);
  }

  /**
   * returns a string representing in .tex.
   * @return a String containing a table entry of a tex table
   */
  public abstract String toTexString();

  /**
   * getter for the lock.
   * @return the lock
   */
  ReentrantReadWriteLock getLock() {
    return this.lock;
  }
}
