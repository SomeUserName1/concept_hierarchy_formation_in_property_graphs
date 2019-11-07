/*
 * @(#)GraphDBConfig.java   1.0   May 15, 2019
 *
 * Copyright (c) 2019- University of Konstanz.
 *
 * This software is the proprietary information of the above-mentioned institutions.
 * Use is subject to license terms. Please refer to the included copyright notice.
 */
package kn.uni.dbis.neo4j.eval.annotations;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;


/**
 * Extension config to the graph database instance used for testing.
 * <p>
 * Usage:
 * <pre><code>
 * {@literal @}Test
 * {@literal @}GraphDBConfig(log = true, readOnly = false, persistent = false)
 * void testMethod(GraphDatabaseService db) {
 *     ...
 * }
 * </code></pre>
 *
 * @author Manuel Hotz &lt;manuel.hotz@uni-konstanz.de&gt;
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.TYPE, ElementType.METHOD})
public @interface GraphDBConfig {

  /**
   * Indicates weather or not to use a user provided logger.
   *
   * @return true if a user provided logger shall be used (see DatabaseFactory.setUserProvidedLog
   */
  boolean log() default true;

  /**
   * indicates weather or not to open the test database instance read only.
   *
   * @return true if the database may be opened read only
   */
  boolean readOnly() default false;

  /**
   * Indicated weather or not to delete the database after testing.
   *
   * @return weather or not to delete the test database.
   */
  boolean persistent() default false;
}
