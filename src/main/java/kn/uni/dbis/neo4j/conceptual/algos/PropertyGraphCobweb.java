package kn.uni.dbis.neo4j.conceptual.algos;

import kn.uni.dbis.neo4j.conceptual.algos.cobweb.COBWEB;
import kn.uni.dbis.neo4j.conceptual.algos.cobweb.ConceptNode;
import kn.uni.dbis.neo4j.conceptual.algos.cobweb.ConceptValue;
import kn.uni.dbis.neo4j.conceptual.algos.cobweb.NominalValue;
import kn.uni.dbis.neo4j.conceptual.algos.cobweb.NumericValue;
import kn.uni.dbis.neo4j.conceptual.algos.cobweb.Value;
import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;

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

        ConceptNode summarizedNode = new ConceptNode();
        Map<Value, Integer> co = new HashMap<>();
        co.put(new ConceptValue(propertyNodes.get(0)), 1);
        summarizedNode.getAttributes().put("NodePropertiesConcept", co);

        co = new HashMap<>();
        for (int i = 1; i < propertyNodes.size(); i++) {
            co.put(new ConceptValue(propertyNodes.get(i)), 1);
        }
        summarizedNode.getAttributes().put("RelationshipConcept", co);

        extractStructuralFeatures(node, summarizedNode);

        cobweb(summarizedNode, this.root, true);
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
     *          AvgNeighbourDegree
     *          NeighbourDegree per RelationshipType
     *          |OutArcsEgoNet|
     *          |InArcsEgoNet|
     *
     *
     * @param node from which to extract the features
     */
    private void extractStructuralFeatures(Node node, ConceptNode conceptNode) {
        int egoDegree = node.getDegree();
        Map<RelationshipType, Integer> egoDegPerType = new HashMap<>();
        Map<RelationshipType, Integer> neighbourDegreePerType = new HashMap<>();
        int totalNeighbourDegree = 0;
        int neighbourDegree;
        RelationshipType relType;
        int noOutArcs = node.getDegree(Direction.OUTGOING);
        int noInArcs = node.getDegree(Direction.INCOMING);

        for (Relationship rel : node.getRelationships()) {
             relType = rel.getType();
            neighbourDegree = rel.getOtherNode(node).getDegree();
            noOutArcs += rel.getOtherNode(node).getDegree(Direction.OUTGOING);
            noInArcs += rel.getOtherNode(node).getDegree(Direction.INCOMING);
            totalNeighbourDegree += neighbourDegree;

            if (!egoDegPerType.containsKey(relType)) {
                egoDegPerType.put(relType, node.getDegree(relType));
            }
            if (neighbourDegreePerType.containsKey(relType)) {
                neighbourDegreePerType.put(relType, neighbourDegreePerType.get(relType) + neighbourDegree);
            } else {
                neighbourDegreePerType.put(relType, neighbourDegree);
            }
        }

        neighbourDegreePerType.replaceAll((k, v) -> v / egoDegPerType.get(k));

        // store features into node
        Map<Value, Integer> temp = new HashMap<>();
        temp.put(new NumericValue(egoDegree), 1);
        conceptNode.getAttributes().put("EgoDegree", temp);

        temp = new HashMap<>();
        temp.put(new NumericValue(totalNeighbourDegree/egoDegree), 1);
        conceptNode.getAttributes().put("AverageNeighbourDegree", temp);

        for (Map.Entry<RelationshipType, Integer> egodegpt : egoDegPerType.entrySet()) {
            temp = new HashMap<>();
            temp.put(new NumericValue(egodegpt.getValue()), 1);
            conceptNode.getAttributes().put(egodegpt.getKey().name() + "_Degree", temp);
        }
        for (Map.Entry<RelationshipType, Integer> neighdegpt : neighbourDegreePerType.entrySet()) {
            temp = new HashMap<>();
            temp.put(new NumericValue(neighdegpt.getValue()), 1);
            conceptNode.getAttributes().put(neighdegpt.getKey().name() + "_NeighbourDegree", temp);
        }

        temp = new HashMap<>();
        temp.put(new NumericValue(noOutArcs), 1);
        conceptNode.getAttributes().put("EgoNetOutgoingEdges", temp);

        temp = new HashMap<>();
        temp.put(new NumericValue(noInArcs), 1);
        conceptNode.getAttributes().put("EgoNetIncomingEdges", temp);
    }

    private void match(ConceptNode toMatch) {
        List<String> rootAttribs = new ArrayList<>(root.getAttributes().keySet());
        List<String> toMatchAttrib = new ArrayList<>(toMatch.getAttributes().keySet());

        double baseEAP = getExpectedAttributePrediction(toMatch);
        double[][] costMatrix = new double[toMatchAttrib.size()][rootAttribs.size()];
        ConceptNode altered;
        int i = 0, j = 0;
        double min;
        int[] minIdx = new int[toMatchAttrib.size()];
        for (String toMatchName : toMatchAttrib) {
            min = 1;
            for (String rootName : rootAttribs) {
                altered = toMatch.clone();
                altered.getAttributes().put(rootName, altered.getAttributes().get(toMatchName));
                altered.getAttributes().remove(toMatchName);
                costMatrix[j][i] = 1 - (COBWEB.getExpectedAttributePrediction(altered) - baseEAP);
                if (costMatrix[j][i] < min) {
                    min = costMatrix[j][i];
                    minIdx[j] = i;
                }
                i++;
            }
            j++;
        }

        // Transform node
        for (j = 0; j < costMatrix.length; j++) {
            if (costMatrix[j][minIdx[j]] < 1) {
                toMatch.getAttributes().put(rootAttribs.get(minIdx[j]), toMatch.getAttributes().get(toMatchAttrib.get(j)));
                toMatch.getAttributes().remove(toMatchAttrib.get(j));
            }
        }
    }
}
