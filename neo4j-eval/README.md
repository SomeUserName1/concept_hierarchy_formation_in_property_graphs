

#### neo4j-eval: Basic utilities testing and benchmarking algorithms with [Neo4J](https://neo4j.com/)  
The code was mainly written by Manuel Hotz in the course of the neo4j-pdsp project. In the following, the check boxes indicate completed refactorization (understanding the single parts, cleaning up the code and project structure).  

#### Features & Requirements  
This package is intended to provide junit annotations for testing __[procedures and functions for the Cypher query 
language](https://neo4j.com/docs/java-reference/3.5/javadocs/index.html?org/neo4j/procedure/Procedure.html)__.  
An example procedure template for creationg new procedures created by the Neo4J team can be found [on GitHub](https://github.com/neo4j-examples/neo4j-procedure-template).  
1. [x] Maven setup of a N4J plugin project.  
2. [x] Database instantiation utilities.  
3. [x] Provide example data sets:  
    1. [x] convert [Stanford network analysis project](http://snap.stanford.edu/data/index.html)'s txt.gz and 
    Challenge9s .gr.gz to csv.  
    2. [x] wrapper classes for the provided datasets  
4. [x] Test utilities:  
    1. [x] Functions to use above example data sets in tests by either importing or copying them using a cypher 
    query, an importer or another database.  
    2. [x] Junit Annotations for time measurements and GraphDatabaseService startup and shutdown in-memory before and
     after each test respectively.  
5. [ ] Tools for benchmarking the custom procedure in terms of runtime, memory footprint, database size.

## Usage: [Basic guide](https://neo4j.com/docs/java-reference/current/extending-neo4j/procedures-and-functions/procedures/)  
1. Create the procedure interface in your project in the src/main/java/<package-name>/proc directory.  
2. Configure the test database and data sets using the test utilities and annotations:  
    - The @GraphDBConfig annotationto specify logging persistance of the test db instance and weather to open it read
     only.
    - The @GraphSource annotation to specify either a Dataset, a query or a path of an existing database folder 
    structure (containing neostore.* files)
    - The @Procedures annotation to specify which procedures to register to cypher in the test db
    - The @Preprocessing annotation to specify Cypher queries to execute before running the actual test. 
3. Implement test cases extending the BaseTest class.
4. Implement algorithm in your project in the src/main/<package-name>/algo directory.  
5. Implement Benchmarks using the RunReport class to get the system specifications and jmh annotations to do the actual benchmark (that is Setup, Teardown and Benchmark/Runner). You can add Profilers using the Runner Options.



# TODOs
- bench: 
    - [ ] add propper profiling utilities
    - [ ] inspect how to macro bench with jmh if possible
- dataset:
    - [ ] Include ldbc snb dataset/integrate shell script with proper parameters
    - [ ] Yelp dataset version objects only and graph based
