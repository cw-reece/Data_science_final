package tasks;

import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
public class FeatureSelector {
    // Method to select important features using InfoGainAttributeEval and Ranker
    public static Instances selectFeatures(Instances data) throws Exception {
        CorrelationAttributeEval eval = new CorrelationAttributeEval();  // Supports numeric classes
        Ranker search = new Ranker();  // Use Ranker for feature selection
        weka.attributeSelection.AttributeSelection selector = new weka.attributeSelection.AttributeSelection();
        selector.setEvaluator(eval);
        selector.setSearch(search);
        selector.SelectAttributes(data);
        return selector.reduceDimensionality(data);
    }
}