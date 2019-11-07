/*
 * @(#)GraphDBSetup.java   1.0   May 15, 2019
 *
 * Copyright (c) 2019- University of Konstanz.
 *
 * This software is the proprietary information of the above-mentioned institutions.
 * Use is subject to license terms. Please refer to the included copyright notice.
 */
package kn.uni.dbis.neo4j.eval.annotations;

import java.io.IOException;
import java.lang.annotation.Annotation;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Optional;

import org.junit.jupiter.api.extension.AfterEachCallback;
import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.ParameterContext;
import org.junit.jupiter.api.extension.ParameterResolver;
import org.junit.platform.commons.support.AnnotationSupport;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.internal.kernel.api.exceptions.KernelException;

import kn.uni.dbis.neo4j.eval.DefaultPaths;
import kn.uni.dbis.neo4j.eval.TestDatabaseFactory;
import kn.uni.dbis.neo4j.eval.datasets.Dataset;
import kn.uni.dbis.neo4j.eval.util.FileUtils;

/**
 * Extension that starts an in-memory instance of {@link GraphDatabaseService} before each test method and shuts it down
 * after each test method.
 * <p>
 * Extensions replace Rules of JUnit4. So that means, we cannot use Neo4JRule from the neo4j-harness.
 * <p>
 * The annotation {@link GraphDBConfig @GraphDBConfig} can be used to configure how the DB is
 * initialized.
 * <p>
 * For example:
 * <pre><code>
 * {@literal @}ExtendWith(GraphDBSetup.class)
 * {@literal @}CypherSource
 * {@literal @}GraphDBConfig(
 *          procedures = {FooProcedure.class, BarProcedure.class},
 *          preprocessing = {Pre1.class, Pre2.class}
 *      )
 * class GraphTestClass {
 *      {@literal @}Test
 *       void testOne(GraphDatabaseService db) {
 *          ...
 *       }
 *       {@literal @}Test
 *       void testTwo(GraphDatabaseService db) {
 *          ...
 *       }
 * }
 * </code></pre>
 *
 * @author Manuel Hotz &lt;manuel.hotz@uni-konstanz.de&gt;
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
@GraphDBConfig
@GraphSource
@Preprocessing
@Procedures
public class GraphDBSetup implements BeforeAllCallback, AfterEachCallback, BeforeEachCallback, ParameterResolver {
  /**
   * Namespaced used to store a reference to the {@link GraphDatabaseService} instance.
   */
  private static final ExtensionContext.Namespace NAMESPACE = ExtensionContext.Namespace.create(GraphDBSetup.class);

  /**
   * Default configuration specified in the {@link GraphDBConfig} annotation interface (log true, rest false).
   */
  private static final GraphDBConfig DEFAULT_CONFIG = GraphDBSetup.class.getAnnotation(GraphDBConfig.class);
  /**
   * Default configuration specified in the {@link GraphSource} annotation interface (EDBT 17 Example).
   */
  private static final GraphSource DEFAULT_SOURCE = GraphDBSetup.class.getAnnotation(GraphSource.class);
  /**
   * Default configuration specified in the {@link Preprocessing} annotation interface (None).
   */
  private static final Preprocessing DEFAULT_PREPROCESSING = GraphDBSetup.class.getAnnotation(Preprocessing.class);
  /**
   * Default configuration specified in the {@link Procedures} annotation interface (None).
   */
  private static final Procedures DEFAULT_PROCEDURES = GraphDBSetup.class.getAnnotation(Procedures.class);


  /**
   * Get the desired annotation from current scope and recursivfly parent scopes if not present in the current.
   *
   * @param ctx  Extention Context of the annotated Test
   * @param clzz Annotation interface.class object
   * @param <T>  Annotation interface type
   * @return the given annotation object
   */
  private <T extends Annotation> Optional<T> getRecursiveAnnotation(final ExtensionContext ctx,
                                                                           final Class<T> clzz) {
    final Optional<T> sourceOption = AnnotationSupport.findAnnotation(ctx.getElement(), clzz);

    if (sourceOption.isPresent()) {
      // Source defined in current context
      return sourceOption;
    } else {
      final Optional<ExtensionContext> parent = ctx.getParent();
      // source definied in parent context
      if (parent.isPresent()) {
        return this.getRecursiveAnnotation(parent.get(), clzz);
      } else {
        // else nothing is defined. Specify default options after call with orElse
        return Optional.empty();
      }
    }
  }

  /**
   * open a graph database instance with the specified source, either from cached, existing or newly setup template
   * instances.
   *
   * @param ctx Extension context to get the source and configuration from
   * @return a {@link GraphDatabaseService} instance to run the test on
   * @throws IOException if cloning the database or opening an existing one or setting up a new template instance fails
   */
  private GraphDatabaseService provideGraphDatabaseService(final ExtensionContext ctx)
      throws IOException {


    final GraphSource source = this.getRecursiveAnnotation(ctx, GraphSource.class).orElse(DEFAULT_SOURCE);
    final Dataset dataset = source.getDataset();
    final String datasetName = dataset.name();

    // Case 1: open a neostore from previous sessions
    if (dataset.getSrcType().equals(Dataset.SourceType.Custom)) {
      final Path src = dataset.getPath();
      if (!Files.isDirectory(src)) {
        throw new IOException("The specified directory does not exist");
      } else if (!Files.isRegularFile(src.resolve("neostore"))) {
        throw new IOException("The specified directory does not contain a neostore!");
      }
      System.out.println("Loading from a previously altered and persisted store!");
      return this.cloneAndOpenGraphDB(ctx, src, datasetName);
    }

    // Case 2 : use cached neostore for the given dataset
    final Iterator<Path> neostores = this.findDatabases();
    Path store;
    while (neostores.hasNext()) {
      store = neostores.next();
      // found an existing neo store to copy
      if (store.toString().contains(datasetName)) {
        System.out.println("USING CACHED NEOSTORE " + store.toString() + " FOR DATASET " + datasetName);
        return this.cloneAndOpenGraphDB(ctx, store, datasetName);
      }
    }
    
    /* Case 3: Download dataset, spawn a new DB and import dataset, shutdown db in order to cache it.
    Use a copy of the above created files.  */
    final Path plainPath = Paths.get(DefaultPaths.PLAIN_STORES_PATH.toString(), datasetName + ".db");
    System.out.println("IMPORTING " + datasetName + " FROM " + source.getDataset().getSource() + " to " + plainPath);

    dataset.fetchAndImport();

    System.out.println("Finished Caching");
    return this.cloneAndOpenGraphDB(ctx, plainPath, datasetName);
  }

  /**
   * Looks in the next 2 depth steps of the file tree for valid database directories.
   *
   * @return directories containing Neo4J databases
   * @throws IOException throws if next was called before hasNext was called and there are no further elements
   */
  private Iterator<Path> findDatabases() throws IOException {
    final Iterator<Path> outer = Files.walk(DefaultPaths.PLAIN_STORES_PATH, 2).iterator();
    return new Iterator<Path>() {

      private Path next = null;
      private boolean closed;

      @Override
      public boolean hasNext() {
        if (this.next != null) {
          return true;
        }
        if (this.closed) {
          return false;
        }
        while (outer.hasNext()) {
          final Path candidate = outer.next();
          if (Files.isDirectory(candidate) && Files.isRegularFile(candidate.resolve("neostore"))) {
            this.next = candidate;
            return true;
          }
        }
        this.closed = true;
        return false;
      }

      @Override
      public Path next() {
        if (!this.hasNext()) {
          throw new NoSuchElementException();
        }
        final Path res = this.next;
        this.next = null;
        return res;
      }
    };
  }

  /**
   * Clone and open the specified template database.
   *
   * @param ctx         Extention context to get the configuration from.
   * @param src         template database to clone
   * @param datasetName name of the database
   * @return a {@link GraphDatabaseService} instance to run the test on
   * @throws IOException if cloning fails
   */
  private GraphDatabaseService cloneAndOpenGraphDB(final ExtensionContext ctx, final Path src,
                                                          final String datasetName) throws IOException {
    final GraphDBConfig config = this.getRecursiveAnnotation(ctx, GraphDBConfig.class).orElse(DEFAULT_CONFIG);

    final TestDatabaseFactory tdf = new TestDatabaseFactory();
    final FileUtils fu = new FileUtils();
    if (!config.persistent()) {
      System.out.println("Source: " + src);
      final Path dst = DefaultPaths.TEMP_STORES_PATH;
      System.out.println("Destination: " + dst);
      fu.clone(src, dst);
      return tdf.openEmbedded(dst.resolve(datasetName + ".db"), config.log(), config.readOnly());
    } else {
      final Path dst = DefaultPaths.PERSISTENT_STORES_PATH.resolve(datasetName + ".db");
      fu.clone(src, dst);
      return tdf.openEmbedded(dst.resolve(datasetName + ".db"), config.log(), config.readOnly());
    }
  }

  @Override
  public void beforeEach(final ExtensionContext ctx) throws IOException, KernelException {

    final GraphDatabaseService db = this.provideGraphDatabaseService(ctx);

    final Procedures procedures = this.getRecursiveAnnotation(ctx, Procedures.class).orElse(DEFAULT_PROCEDURES);
    final TestDatabaseFactory tdf = new TestDatabaseFactory();
    tdf.registerProcedure(db, procedures.procedures());

    final Preprocessing preproc = this.getRecursiveAnnotation(ctx, Preprocessing.class).orElse(DEFAULT_PREPROCESSING);
    for (final String pre : preproc.preprocessing()) {
      if (!pre.isEmpty()) {
        db.execute(pre).close();
      }
    }
    ctx.getStore(NAMESPACE).put(GraphDatabaseService.class, db);
    Runtime.getRuntime().addShutdownHook(new Thread(db::shutdown));
    new FileUtils().recursiveDeleteOnExit(DefaultPaths.TEMP_HOME_PATH);
    final GraphSource source = this.getRecursiveAnnotation(ctx, GraphSource.class).orElse(DEFAULT_SOURCE);
    ctx.getStore(NAMESPACE).put(Dataset.class, source.getDataset());
  }

  @Override
  public void afterEach(final ExtensionContext ctx) {
    final GraphDatabaseService db = ctx.getStore(NAMESPACE).get(GraphDatabaseService.class, GraphDatabaseService.class);
    db.shutdown();
  }

  @Override
  public void beforeAll(final ExtensionContext ctx) {
    final GraphDBConfig config = this.getRecursiveAnnotation(ctx, GraphDBConfig.class).orElse(DEFAULT_CONFIG);
    final GraphSource source = this.getRecursiveAnnotation(ctx, GraphSource.class).orElse(DEFAULT_SOURCE);
    ctx.getStore(NAMESPACE).put(GraphDBConfig.class, config);
    ctx.getStore(NAMESPACE).put(GraphSource.class, source);


    if (!DefaultPaths.TEMP_STORES_PATH.toFile().exists() && !DefaultPaths.TEMP_STORES_PATH.toFile().mkdirs()
        || !DefaultPaths.PLAIN_STORES_PATH.toFile().exists() && !DefaultPaths.PLAIN_STORES_PATH.toFile().mkdirs()
        || !DefaultPaths.PERSISTENT_STORES_PATH.toFile().exists()
        && !DefaultPaths.PERSISTENT_STORES_PATH.toFile().mkdirs()) {
      System.out.println("Couldn't create some of the resource directories!");
    }

    // final String path = ClassLoader.getSystemResource("logging.properties").getPath();
    // System.setProperty("java.util.logging.config.file", path);
  }

  @Override
  public boolean supportsParameter(final ParameterContext pCtx, final ExtensionContext ctx) {
    final Class<?> cls = pCtx.getParameter().getType();
    return cls.equals(GraphDatabaseService.class) || cls.equals(Dataset.class);
  }

  @Override
  public Object resolveParameter(final ParameterContext pCtx, final ExtensionContext ctx) {
    // this should be our GraphDBConfig class
    final Class<?> configClass = pCtx.getParameter().getType();
    return ctx.getStore(NAMESPACE).get(configClass, configClass);
  }
}
