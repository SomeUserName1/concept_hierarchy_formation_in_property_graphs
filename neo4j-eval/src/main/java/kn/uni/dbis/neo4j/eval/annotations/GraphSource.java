/*
 * @(#)GraphSource.java   1.0   May 15, 2019
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

import kn.uni.dbis.neo4j.eval.datasets.Dataset;

/**
 * Source database using importing data with an importer.
 *
 * @author Manuel Hotz &lt;manuel.hotz@uni-konstanz.de&gt;
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
@Target({ElementType.TYPE, ElementType.METHOD, ElementType.ANNOTATION_TYPE})
@Retention(RetentionPolicy.RUNTIME)
public @interface GraphSource {
  /**
   * File path to import from.
   *
   * @return file path
   */
  Dataset getDataset() default Dataset.EDBT17_RUNNING_EXAMPLE;
}
