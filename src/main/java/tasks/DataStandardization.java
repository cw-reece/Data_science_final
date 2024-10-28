package tasks;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
public class DataStandardization {

    public static Instances standardize(Instances data) throws Exception {
        Standardize standardizeFilter = new Standardize();
        standardizeFilter.setInputFormat(data);
        return Filter.useFilter(data, standardizeFilter);
    }
}