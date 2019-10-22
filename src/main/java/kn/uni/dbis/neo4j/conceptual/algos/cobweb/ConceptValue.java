package kn.uni.dbis.neo4j.conceptual.algos.cobweb;

public class ConceptValue implements Value {
    private ConceptNode concept;

    ConceptValue(ConceptNode node) {
        this.concept = node;
    }

    @Override
    public Value clone() {
        return new ConceptValue(this.concept.clone());
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof ConceptNode) {
            return ((ConceptNode)o).equals(this.concept);
        }
    }
}
