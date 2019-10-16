package kn.uni.dbis.neo4j.conceptual.algos.cobweb;

import org.neo4j.graphdb.Node;

import java.util.Map;

public class COBWEB {
    protected ConceptNode root;
    private static double cont_cu_const = 4*Math.sqrt(Math.PI);

    public COBWEB() {
        this.root = new ConceptNode();
    }

    public void integrate(Node node) {
        ConceptNode newChild = new ConceptNode();
        newChild.nodePropertiesToConcept(node);
        cobweb(newChild, this.root, true);
    }

    protected static void cobweb(ConceptNode newNode, ConceptNode currentNode, boolean updateCurrent) {
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

    private static double computeCU(ConceptNode parent, double parentEAP) {
        double cu = 0.0;
        double parentCount = parent.getCount();
        for (ConceptNode child : parent.getChildren()) {
            cu += (double)child.getCount()/parentCount
                    * ( getExpectedAttributePrediction(child) - parentEAP);
        }
        return cu/(double)parent.getChildren().size();
    }


    public static double getExpectedAttributePrediction(ConceptNode category) {
        double exp = 0;
        double total = category.getCount();
        for (Map.Entry<String, Map<Value, Integer>> attrib : category.getAttributes().entrySet()) {
            for (Map.Entry<Value, Integer> val : attrib.getValue().entrySet()) {
                if (val.getKey() instanceof NominalValue) {
                    exp += ((double) val.getValue() / total) * ((double) val.getValue() / total);
                } else {
                    exp += 1.0/(((NumericValue)val.getKey()).getStd() * cont_cu_const);
                }
            }
        }
        return exp/category.getAttributes().size();
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