package kn.uni.dbis.neo4j.conceptual.algos;

import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PropertyGraphCobwebImpl {
    private PropertyGraphCobwebImpl() {
        // NOOP
    }

    public static void integrate(Node node) {
        ConceptNode root = new ConceptNode();
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
            cobweb(cNode, root, true);
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

        cobweb(summarizedNode, root, true);
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
    private static void extractStructuralFeatures(Node node, ConceptNode conceptNode) {
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

    // FIXME double check
    private void match(ConceptNode toMatch, ConceptNode root) {
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
                costMatrix[j][i] = 1 - (PropertyGraphCobwebImpl.getExpectedAttributePrediction(altered) - baseEAP);
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


    private static void cobweb(ConceptNode newNode, ConceptNode currentNode, boolean updateCurrent) {
        if (updateCurrent) {
            currentNode.updateCounts(newNode, false);
        }
        if (currentNode.getChildren().isEmpty()) {
            currentNode.addChild(newNode);
        } else {
            OpResult[] results = new OpResult[4];

            results[0] = findHost(currentNode, newNode);
            ConceptNode host = results[0].getNode();

            results[1] = createNewNode(currentNode, newNode);
            results[2] = mergeNodes(currentNode, host, newNode);
            results[3] = splitNodes(host, newNode);

            OpResult best = results[0];
            for (OpResult result : results) {
                if (result.getCU() > best.getCU()) {
                    best = result;
                }
            }

            switch (best.operation) {
                case CREATE:
                    newNode.setParent(currentNode);
                    currentNode.addChild(newNode);
                    break;
                case SPLIT:
                    for (ConceptNode child : host.getChildren()) {
                        currentNode.addChild(child);
                    }
                    currentNode.getChildren().remove(host);
                    cobweb(newNode, currentNode, false);
                    break;
                case MERGE:
                    currentNode.getChildren().remove(host);
                    currentNode.getChildren().add(results[2].getNode());
                    cobweb(newNode, results[2].getNode(), true);
                    break;
                case RECURSE:
                    cobweb(newNode, host, true);
                    break;
            }
        }
    }

    private static OpResult findHost(ConceptNode parent, ConceptNode newNode) {
        double curCU;
        double maxCU = -1;
        int i = 0;
        ConceptNode clone;
        ConceptNode best = parent;
        ConceptNode parentClone;
        final double parentEAP = getExpectedAttributePrediction(parent);

        for (ConceptNode child : parent.getChildren()) {
            clone = child.clone();
            clone.updateCounts(newNode, false);
            parentClone = parent.clone();
            parentClone.getChildren().set(i, clone);
            curCU = computeCU(parentClone, parentEAP);
            if (maxCU < curCU) {
                maxCU = curCU;
                best = child;
            }
            i++;
        }
        return new OpResult(Op.RECURSE, maxCU, best);
    }

    private static OpResult createNewNode(ConceptNode currentNode, ConceptNode newNode) {
        ConceptNode clone = currentNode.clone();
        clone.addChild(newNode);
        return new OpResult(Op.CREATE, computeCU(clone), clone);
    }

    private static OpResult splitNodes(ConceptNode host, ConceptNode current) {
        ConceptNode currentClone = current.clone();
        for (ConceptNode child : host.getChildren()) {
            currentClone.addChild(child);
        }
        currentClone.getChildren().remove(host);
        return new OpResult(Op.SPLIT, computeCU(current), currentClone);
    }

    private static OpResult mergeNodes(ConceptNode current, ConceptNode host, ConceptNode newNode) {
        ConceptNode clonedParent = current.clone();
        clonedParent.getChildren().remove(host);
        OpResult secondHost = findHost(clonedParent, newNode);
        ConceptNode mNode = host.clone();
        mNode.updateCounts(secondHost.getNode(), true);
        mNode.addChild(host);
        mNode.addChild(secondHost.getNode());
        clonedParent.getChildren().add(mNode);
        return new OpResult(Op.MERGE, computeCU(clonedParent), mNode);
    }


    private static double computeCU(ConceptNode parent) {
        double cu = 0.0;
        final double parentEAP = getExpectedAttributePrediction(parent);
        double parentCount = parent.getCount();
        for (ConceptNode child : parent.getChildren()) {
            cu += (double)child.getCount()/parentCount
                    * ( getExpectedAttributePrediction(child) - parentEAP);
        }
        return cu/(double)parent.getChildren().size();
    }

    // FIXME double check
    private static double computeCU(ConceptNode parent, double parentEAP) {
        double cu = 0.0;
        double parentCount = parent.getCount();
        for (ConceptNode child : parent.getChildren()) {
            cu += (double)child.getCount()/parentCount
                    * ( getExpectedAttributePrediction(child) - parentEAP);
        }
        return cu/(double)parent.getChildren().size();
    }


    private static double getExpectedAttributePrediction(ConceptNode category) {
        double exp = 0;
        double total = category.getCount();
        double interm;

        for (Map.Entry<String, Map<Value, Integer>> attrib : category.getAttributes().entrySet()) {
            for (Map.Entry<Value, Integer> val : attrib.getValue().entrySet()) {
                interm = 0;
                if (val.getKey() instanceof NominalValue) {
                    interm = (double) val.getValue() / total;
                    exp +=  interm * interm;
                } else if (val.getKey() instanceof NumericValue) {
                    exp += 1.0/((NumericValue)val.getKey()).getStd();
                } else if (val.getKey() instanceof ConceptValue) {
                    ConceptValue con = (ConceptValue) val.getKey();
                    for (Map.Entry<Value, Integer> cVal : attrib.getValue().entrySet()) {
                        interm += con.getFactor((ConceptValue)cVal.getKey()) * cVal.getValue()/ total;
                    }
                    exp += interm * interm;
                }

            }
        }
        return exp;
    }

    enum Op {
        CREATE,
        SPLIT,
        MERGE,
        RECURSE
    }

    private static class OpResult {
        private Op operation;
        private double cu;
        private ConceptNode node;

        private OpResult(Op operation, double cu, ConceptNode node) {
            this.operation = operation;
            this.cu = cu;
            this.node = node;
        }

        private ConceptNode getNode() {
            return this.node;
        }

        private double getCU() {
            return this.cu;
        }
    }
}