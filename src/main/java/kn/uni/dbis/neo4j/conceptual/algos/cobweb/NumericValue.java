package kn.uni.dbis.neo4j.conceptual.algos.cobweb;

class NumericValue implements Value {
    private double mean;
    private double std;

    NumericValue(final Number nr) {
        this.mean = nr.doubleValue();
        this.std = 0.0f;
    }

    public double getMean() {
        return this.mean;
    }

    public void setMean(final double mean) {
        this.mean = mean;
    }

    public double getStd() {
        return this.std;
    }

    public void setStd(final double std) {
        this.std = std;
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof NumericValue) {
            NumericValue nr = (NumericValue)o;
            double min_difference = this.mean -  nr.getMean() > 0 ? this.mean - 3*this.std - nr.mean + 3*nr.std :
                    nr.mean - 3*nr.std - this.mean + 3*this.std;
            return min_difference <= 0;
        } else if (o instanceof Number) {
            double nr = ((Number) o).doubleValue();
            return nr > this.mean ? nr - this.mean + 3*this.std <= 0 : this.mean - 3*this.std - nr <= 0;
        } else {
            return false;
        }
    }
}
