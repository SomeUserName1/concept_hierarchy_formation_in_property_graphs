package kn.uni.dbis.neo4j.conceptual.algos.trestle;

import kn.uni.dbis.neo4j.conceptual.algos.cobweb.COBWEB;
import kn.uni.dbis.neo4j.conceptual.algos.cobweb.ConceptNode;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

class NodeMatcher {
    private Iterator<String> rootNames;
    private List<String> newNames;
    private List<Match> openList;
    private List<Match> closedList;
    private double baseEAP;
    private ConceptNode toMatch;


    NodeMatcher(ConceptNode toMatch, ConceptNode root) {
        double[][] costMatrix = new double[toMatch.getAttributes().size()][root.getAttributes().size()];
        ConceptNode altered;
        int i = 0, j = 0;
        double min = -1;
        int[] minIdx = new int[toMatch.getAttributes().size()];
        for (String toMatchName : toMatch.getAttributes().keySet()) {
            for (String rootName : root.getAttributes().keySet()) {
                altered = toMatch.clone();
                altered.getAttributes().put(rootName, altered.getAttributes().get(toMatchName));
                altered.getAttributes().remove(toMatchName);
                costMatrix[j][i] = 1 - (COBWEB.getExpectedAttributePrediction(altered) - this.baseEAP);
                if (costMatrix[j][i] < min) {
                    min = costMatrix[j][i];
                    minIdx[j] = i;
                }
                i++;
            }
            j++;
        }
        // subtract row
        for (j = 0; j < costMatrix.length; j++) {
            for (i = 0; i < costMatrix[j].length; i++) {
                costMatrix[j][i] = costMatrix[j][i] - costMatrix[j][minIdx[j]];
            }
        }
        // TODO do transormation; consider adding a no change field and match for all fields at least one
    }


    public void hungarian() {
        double[][] costMatrix = new double[][]
    }

    public ConceptNode match() {
        expand(new Match(null, null, null, 0));

        Match maxGain;
        double max;
        do {
            max = 0;
            maxGain = null;
            for (Match openNode : openList) {
                if (openNode.fVal >= max) {
                    maxGain = openNode;
                }
            }
            if (maxGain == null) {
                if (this.rootNames.hasNext()) {
                    expand(new Match(null, null, null, 0));
                } else {
                    return this.toMatch;
                }
            } else {
                openList.remove(maxGain);
                closedList.add(maxGain);
                expand(maxGain);
            }
        } while (!openList.isEmpty());

        return recurseReconstruct(this.closedList.get(closedList.size() - 1));
    }

    private void expand(Match nextMatch) {
        String nextName = this.rootNames.hasNext() ? rootNames.next() : null;
        if (nextName == null) {
            return;
        }
        List<String> predNames = getPredecessorUsedNames(nextMatch);
        double gVal;
        ConceptNode altered;
        Match m;
        for (String child : this.newNames) {
            if (predNames.contains(child) || closedList.contains(nextMatch)) {
                continue;
            }
            altered = toMatch.clone();
            altered.getAttributes().put(nextName, altered.getAttributes().get(child));
            altered.getAttributes().remove(child);
            gVal = COBWEB.getExpectedAttributePrediction(altered) - this.baseEAP;
            if (gVal < 0 ) {
                continue;
            }
            m = new Match(nextName, child, nextMatch, gVal);

            if (openList.contains(m)) {
                Match otherM = openList.get(openList.indexOf(m));
                if (otherM.fVal < gVal) {
                    openList.remove(otherM);
                    openList.add(m);
                }
            } else {
                openList.add(m);
            }
        }
        // no changes
        openList.add(new Match(null, null, nextMatch, COBWEB.getExpectedAttributePrediction()));
    }

    private static List<String> getPredecessorUsedNames(Match pred) {
        List<String> result = new ArrayList<>();
        if (pred.newChildAttribute != null) {
            result.add(pred.newChildAttribute);
        }
        if (pred.predecessor != null) {
            result.addAll(getPredecessorUsedNames(pred.predecessor));
        }
        return result;
    }

    private ConceptNode recurseReconstruct(Match m) {
        if (m.rootAttribute != null) {
            toMatch.getAttributes().put(m.rootAttribute, toMatch.getAttributes()
                    .get(m.newChildAttribute));
            toMatch.getAttributes().remove(m.newChildAttribute);
        }
        if (m.predecessor != null) {
            recurseReconstruct(m.predecessor);
        }
        return toMatch;
    }

    private static class Match {
        String rootAttribute;
        String newChildAttribute;
        Match predecessor;
        double fVal;

        Match(String rootAttrib, String newNodeAttrib, Match pred, double gain) {
            this.rootAttribute = rootAttrib;
            this.newChildAttribute = newNodeAttrib;
            this.fVal = gain;
            this.predecessor = pred;
        }

        @Override
        public boolean equals(Object o) {
            if ( o instanceof Match) {
                Match m = (Match)o;
                return m.rootAttribute.equals(this.rootAttribute) && m.newChildAttribute.equals(this.newChildAttribute);
            } else {
                return false;
            }
        }
    }
}
