package tasks;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
public class DataNormalization {
    public static Instances normalize(Instances data) throws Exception {
        Normalize normalizeFilter = new Normalize();
        normalizeFilter.setInputFormat(data); // Configure the filter with the input data
        return Filter.useFilter(data, normalizeFilter);
    }

}
