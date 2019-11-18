package kn.uni.dbis.neo4j.conceptual.util;

/**
 * Helper Interface for an AutoClosable lock.
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public interface ResourceLock extends AutoCloseable {

  /**
   * Unlocking doesn't throw any checked exception.
   */
  @Override
  void close();
}
