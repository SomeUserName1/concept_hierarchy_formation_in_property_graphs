package kn.uni.dbis.neo4j.conceptual.algos;

public class ConceptValue implements Value {
    private ConceptNode concept;

    ConceptValue(ConceptNode node) {
        this.concept = node;
    }

    @Override
    public Value clone() {
        try {
            super.clone();
        } catch (final CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return new ConceptValue(this.concept.clone());
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof ConceptNode) {
            return this.concept.equals(o);
        } else if (o instanceof ConceptValue) {
            return this.concept.equals(o);
        } else {
            return false;
        }
    }

    // Conditional probability of having concept c_i when given c_j bzw P(this|val)
    double getFactor(ConceptValue val) {
        if (val.concept.equals(this.concept) || this.concept.isSuperConcept(val.concept)) {
            return 1.0;
        } else if (val.concept.isSuperConcept(this.concept)) {
            return (double)this.concept.getCount()/(double)val.concept.getCount();
        } else {
            // they're on different paths => disjoint
            return 0;
        }
    }
}
