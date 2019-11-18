package kn.uni.dbis.neo4j.conceptual.util;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Auto-Closeable Multi Object Locking.
 *
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public class LockUtils {
  private LockUtils() {
    // NOOP
  }

  public static ResourceLock lockAll(List<Lock> locks) {
    return lockAll(locks.toArray(new Lock[0]));
  }

  public static ResourceLock lockAll(Lock... locks) {
    List<Lock> successful = new ArrayList<>();
    boolean acquired = false;

    for (final Lock lock : locks) {
      acquired = false;
      try {
        acquired = lock.tryLock(3, TimeUnit.SECONDS);
      } catch (InterruptedException e) {
        e.printStackTrace();
      }
      if (acquired) {
        successful.add(lock);
      } else {
        break;
      }
    }
    if (!acquired) {
      for (Lock lock1 : successful) {
        lock1.unlock();
      }
      // Preempt the thread and try again
      Thread.yield();
      return lockAll(locks);
    }

    return  () -> {
      for (final Lock lock : locks) {
        lock.unlock();
      }
    };
  }
}


