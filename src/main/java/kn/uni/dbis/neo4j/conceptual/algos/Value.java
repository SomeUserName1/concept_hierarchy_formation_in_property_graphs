package kn.uni.dbis.neo4j.conceptual.algos;

// Them Neo4j retards force one to do this instance of shit due to their "non explosion of overload" shit


public interface Value {
    Value clone();
    boolean equals(Object o);
    static Value cast(Object o) {
        if (o instanceof String) {
            return new NominalValue((String)o);
        } else if (o instanceof Boolean) {
            return new NominalValue((boolean)o);
        } else if (o instanceof Character) {
            return new NominalValue((Character)o);
        } else if (o instanceof Number) {
            return new NumericValue((Number)o);
        } else {
            System.out.println("Encountered Property with unsupported type!");
            return new NominalValue("Unsupported Type!");
        }
    }
}
