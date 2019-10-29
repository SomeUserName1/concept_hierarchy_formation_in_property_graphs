package kn.uni.dbis.neo4j.conceptual.algos;

// Them Neo4j retards force one to do this instance of shit due to their "non explosion of overload" shit

/**
 * Interface for a Value to cluster using Cobweb.
 * I.e. it's held by the attributes map as value in the @link{ConceptNode} class.
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public interface Value {
    /**
     * Used when computing the CU for the 4 operations of cobweb (insert, split, merge, recurse).
     * @return returns a deep copy of the Value it's called on.
     */
    Value clone();

    /**
     * unpacks the held value and checks for equality depending on the ValueType (e.g. NominalValue checks
     * for String equivalence, NumericValue for overlap in the 3sigma space arround the means.
     * @param o object to compare
     * @return a boolean inidcating equivalence
     */
    boolean equals(final Object o);

    /**
     * given a primitive datatype or a String, decide which Value type to instantiate.
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
        } else if (o instanceof ConceptNode) {
            return new ConceptValue((ConceptNode) o);
        } else {
            System.out.println("Encountered Property with unsupported type!");
            return new NominalValue("Unsupported Type!");
        }
    }
}
