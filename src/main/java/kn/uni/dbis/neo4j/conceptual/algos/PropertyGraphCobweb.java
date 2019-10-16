package kn.uni.dbis.neo4j.conceptual.algos;

import kn.uni.dbis.neo4j.conceptual.algos.cobweb.COBWEB;
import kn.uni.dbis.neo4j.conceptual.algos.cobweb.ConceptNode;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class PropertyGraphCobweb extends COBWEB {
    PropertyGraphCobweb() {
        super();
    }

    @Override
    public void integrate(Node node) {

        // Static categorization according to properties
        // TODO handle sets in cobweb. Maybe Attribute = A,1; b,1; c,1 for each occurence?
        List<ConceptNode> propertyNodes = new ArrayList<>();
        ConceptNode nodeProperties = new ConceptNode();
        nodeProperties.propertyContainerToConceptNode(node);
        propertyNodes.add(nodeProperties);

        ConceptNode relProperties;
        for (Relationship rel : node.getRelationships()) {
            relProperties = new ConceptNode();
            relProperties.propertyContainerToConceptNode(rel);
            propertyNodes.add(relProperties);
        }

        for (ConceptNode cNode : propertyNodes) {
            cobweb(cNode, this.root, true);
        }

        // TODO restructure node according to results from cobweb

    }

    public void extractStructuralFeatures(Node node) {
        int degree = node.getDegree();
        Map<RelationshipType, Integer> degPerType = new HashMap<>();
        for (Relationship rel : node.getRelationships()) {
            RelationshipType relType = rel.getType();
            if (degPerType.containsKey(relType)) {
                degPerType.put(relType, degPerType.get(relType) + 1);
            } else {
                degPerType.put(relType, 1);
            }
        }
        // TODO Extract structure features here
        // TODO add to node as properties
    }

    public void extractLabelBasedFeatures(Node node) {

    }
}
