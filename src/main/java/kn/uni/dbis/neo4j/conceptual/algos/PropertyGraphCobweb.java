package kn.uni.dbis.neo4j.conceptual.algos;

import kn.uni.dbis.neo4j.conceptual.algos.cobweb.COBWEB;
import kn.uni.dbis.neo4j.conceptual.algos.cobweb.ConceptNode;
import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;

import javax.management.relation.Relation;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PropertyGraphCobweb extends COBWEB {
    PropertyGraphCobweb() {
        super();
    }

    @Override
    public void integrate(Node node) {
        // Static categorization according to properties, labels and relationship type
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
        // TODO how to get back the result of cobweb?
        //  returning the parent where the node was sorted?
        //  how to get the mapping? nodeids/relationshipids are considered bad practice
        ConceptNode summarizedNode = new ConceptNode();
        summarizedNode.getAttributes().put()
        extractStructuralFeatures(node);
        // TODO restructure node according to results from cobweb

    }


    /**
     * Desired features to be extracted:
     * [x] EgoDegree
     * [x] EgoDegreePerType
     * [x] AvgNeighbourDegree
     * [x] AvgNeighbourDegreePerType
     *
     * Resulting ConceptNode structure:
     * ConceptNode
     *      *Label-based*
     *          NodeConcept (property-based)
     *          RelationshipConcepts (property-based)
     *
     *      *StructureBased*
     *          EgoDegree
     *          EgoDeg per RelationshipType (Characteristic set with counts)
     *          CountNeighbourDegree/|EgoDegree|
     *          NeighbourDegree per RelationshipType
     *          |ArcsEgoNet|
     *
     *
     * @param node
     */
    public void extractStructuralFeatures(Node node) {
        int egoDegree = node.getDegree();
        Map<RelationshipType, Integer> egoDegPerType = new HashMap<>();
        Map<RelationshipType, Integer> neighbourDegreePerType = new HashMap<>();
        int totalNeighbourDegree = 0;
        int neighbourDegree;
        RelationshipType relType;
        int noArcs = node.getDegree(Direction.OUTGOING);

        for (Relationship rel : node.getRelationships()) {
             relType = rel.getType();
            if (!egoDegPerType.containsKey(relType)) {
                egoDegPerType.put(relType, node.getDegree(relType));
            }
            neighbourDegree = rel.getOtherNode(node).getDegree();
            totalNeighbourDegree += neighbourDegree;
            neighbourDegreePerType.put(relType, neighbourDegreePerType.get(relType) + neighbourDegree);
            noArcs += rel.getOtherNode(node).getDegree(Direction.OUTGOING);
        }
        int avgNeighbourDegree = totalNeighbourDegree/egoDegree;

        for (Map.Entry<RelationshipType, Integer> entry : neighbourDegreePerType.entrySet()) {
            neighbourDegreePerType.put(entry.getKey(), entry.getValue()/egoDegPerType.get(entry.getKey()));
        }


    }

    public void extractLabelBasedFeatures(Node node) {
        // TODO addition to guide role extraction; stick with structural for now
    }
}
