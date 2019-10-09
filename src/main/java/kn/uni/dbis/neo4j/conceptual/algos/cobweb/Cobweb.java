package kn.uni.dbis.neo4j.conceptual.algos.cobweb;

import com.rits.cloning.Cloner;
import org.neo4j.graphdb.Node;

import java.util.Map;

class Cobweb {
    private ConceptNode root;
    private double maxCU;

    Cobweb() {
        this.root = new ConceptNode();
        this.maxCU = -1.0;
    }

    public void integrate(Node node) {
        ConceptNode newChild = new ConceptNode();
        newChild.nodePropertiesToConcept(node);
        cobweb(newChild, this.root, true);
    }

    public void cobweb(ConceptNode newNode, ConceptNode currentNode, boolean updateCurrent) {
        currentNode.updateCounts(newNode);

        if (currentNode.getChildren().isEmpty()) {
            currentNode.addChild(newNode);
        } else {
            ConceptNode host = findHost(currentNode, newNode);
            double createCU = createNewNodeCU(currentNode, newNode);
            double mergeCU = mergeNodesCU(host, newNode);
            double splitCU = splitNodesCU(host, newNode);

            if (createCU > mergeCU && createCU > splitCU && createCU > this.maxCU) {
                createNewNode(currentNode, newNode);
            } else if (mergeCU > splitCU && mergeCU > this.maxCU) {
                mergeNodes(host, newNode);
                cobweb(newNode, this.root, false);
            } else if (splitCU > this.maxCU) {
                splitNodes(host, currentNode);
                cobweb(newNode, this.root, false);
            } else {
                cobweb(newNode, host, true);
            }
        }
    }

    private void splitNodes(ConceptNode host, ConceptNode currentNode) {
        for (ConceptNode child : host.getChildren()) {
            currentNode.addChild(child);
        }
        currentNode.getChildren().remove(host);
    }

    private void mergeNodes(ConceptNode host, ConceptNode newNode) {
        // TODO
    }

    private ConceptNode findHost(ConceptNode parent, ConceptNode newNode) {
        Cloner cloner = new Cloner();
        double curCU;
        this.maxCU = -1;
        int i = 0;
        ConceptNode clone;
        ConceptNode best = parent;
        ConceptNode parentClone;

        for (ConceptNode child : parent.getChildren()) {
            clone = cloner.deepClone(child);
            clone.updateCounts(newNode);
            parentClone = cloner.deepClone(parent);
            parentClone.getChildren().set(i, clone);
            curCU = computeCU(parentClone);
            if (this.maxCU < curCU) {
                this.maxCU = curCU;
                best = clone;
            }
            i++;
        }
        return best;
    }

    private double createNewNodeCU(ConceptNode currentNode, ConceptNode newNode) {
        Cloner cloner = new Cloner();
        double cu = -1;
        ConceptNode clone = cloner.deepClone(currentNode);
        clone.addChild(newNode);
        return computeCU(clone);
    }

    private void createNewNode(ConceptNode currentNode, ConceptNode newNode) {
        currentNode.addChild(newNode);
    }

    private double splitNodesCU(ConceptNode host, ConceptNode current) {
        Cloner cloner = new Cloner();
        ConceptNode currentClone = cloner.deepClone(current);
        for (ConceptNode child : host.getChildren()) {
            currentClone.addChild(child);
        }
        currentClone.getChildren().remove(host);
        return computeCU(current);
    }

    private double mergeNodesCU(ConceptNode host, ConceptNode newNode) {
        // TODO
        return 0;
    }

    private double computeCU(ConceptNode parent) {
        double cu = 0.0;

        for (ConceptNode child : parent.getChildren()) {

            cu += (double)child.getCount()/(double)parent.getCount()
                    * this.getExpectedAttributePrediction(child) - this.getExpectedAttributePrediction(parent);
        }
        return cu/(double)parent.getChildren().size();
    }

    private double getExpectedAttributePrediction(ConceptNode categroy) {
        double exp = 0;
        double total = categroy.getCount();
        for (Map.Entry<String, Map<Value, Integer>> attrib : categroy.getAttributes().entrySet()) {
            for (Map.Entry<Value, Integer> val : attrib.getValue().entrySet()) {
                exp += (double)val.getValue()/total;
            }
        }
        return exp;
    }
}