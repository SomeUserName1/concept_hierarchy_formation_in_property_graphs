package kn.uni.dbis.neo4j.conceptual.algos.cobweb;

import org.neo4j.cypher.internal.v3_4.functions.Pi;
import org.neo4j.graphdb.Node;

import java.util.Map;

class COBWEB {
    private ConceptNode root;

    private COBWEB() {
        this.root = new ConceptNode();
    }

    public void integrate(Node node) {
        ConceptNode newChild = new ConceptNode();
        newChild.nodePropertiesToConcept(node);
        cobweb(newChild, this.root, true);
    }

    public void cobweb(ConceptNode newNode, ConceptNode currentNode, boolean updateCurrent) {
        currentNode.updateCounts(newNode, false);

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

    private OpResult findHost(ConceptNode parent, ConceptNode newNode) {
        double curCU;
        double maxCU = -1;
        int i = 0;
        ConceptNode clone;
        ConceptNode best = parent;
        ConceptNode parentClone;

        for (ConceptNode child : parent.getChildren()) {
            clone = child.clone();
            clone.updateCounts(newNode, false);
            parentClone = parent.clone();
            parentClone.getChildren().set(i, clone);
            curCU = computeCU(parentClone);
            if (maxCU < curCU) {
                maxCU = curCU;
                best = child;
            }
            i++;
        }
        return new OpResult(Op.RECURSE, maxCU, best);
    }

    private OpResult createNewNode(ConceptNode currentNode, ConceptNode newNode) {
        ConceptNode clone = currentNode.clone();
        clone.addChild(newNode);
        return new OpResult(Op.CREATE, computeCU(clone), clone);
    }

    private OpResult splitNodes(ConceptNode host, ConceptNode current) {
        ConceptNode currentClone = current.clone();
        for (ConceptNode child : host.getChildren()) {
            currentClone.addChild(child);
        }
        currentClone.getChildren().remove(host);
        return new OpResult(Op.SPLIT, computeCU(current), currentClone);
    }

    private OpResult mergeNodes(ConceptNode current, ConceptNode host, ConceptNode newNode) {
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


    private double computeCU(ConceptNode parent) {
        double cu = 0.0;
        for (ConceptNode child : parent.getChildren()) {
            cu += (double)child.getCount()/(double)parent.getCount()
                    * ( this.getExpectedAttributePrediction(child) - this.getExpectedAttributePrediction(parent));
        }
        return cu/(double)parent.getChildren().size();
    }

    private double getExpectedAttributePrediction(ConceptNode categroy) {
        double exp = 0;
        double total = categroy.getCount();
        for (Map.Entry<String, Map<Value, Integer>> attrib : categroy.getAttributes().entrySet()) {
            for (Map.Entry<Value, Integer> val : attrib.getValue().entrySet()) {
                if (val.getKey() instanceof NominalValue) {
                    exp += ((double) val.getValue() / total) * ((double) val.getValue() / total);
                } else {
                    exp += 1.0/(((NumericValue)val.getKey()).getStd() * 4 * Math.PI);
                }
            }
        }
        return exp/categroy.getAttributes().size();
    }

    enum Op {
        CREATE,
        SPLIT,
        MERGE,
        RECURSE
    }

    private class OpResult {
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