package tasks;

import weka.core.Instances;

import java.util.List;

public class AirQualityFeatureEngineering {

    // Applies feature engineering to the dataset: Rolling averages, lagged values, etc.
    public static void apply(Instances data, List<String> pollutants) {
        for (String pollutant : pollutants) {
            int index = data.attribute(pollutant).index();
            if (index == -1) {
                System.err.println("Error: Pollutant attribute '" + pollutant + "' not found.");
                continue;
            }

            // Add 7-day and 30-day rolling averages
            addRollingAverage(data, index, pollutant, 7);
            addRollingAverage(data, index, pollutant, 30);
        }
    }

    // Adds a rolling average for the specified pollutant
    private static void addRollingAverage(Instances data, int index, String pollutant, int windowSize) {
        String attrName = pollutant + "_rolling_" + windowSize + "d";
        if (data.attribute(attrName) != null) {
            System.out.println("Attribute '" + attrName + "' already exists. Skipping.");
            return;
        }

        data.insertAttributeAt(new weka.core.Attribute(attrName), data.numAttributes());
        int newAttrIndex = data.numAttributes() - 1;

        for (int i = 0; i < data.numInstances(); i++) {
            if (i >= windowSize - 1) {
                double sum = 0.0;
                int count = 0;

                for (int j = i; j > i - windowSize; j--) {
                    if (!data.instance(j).isMissing(index)) {
                        sum += data.instance(j).value(index);
                        count++;
                    }
                }

                if (count > 0) {
                    data.instance(i).setValue(newAttrIndex, sum / count);
                } else {
                    data.instance(i).setMissing(newAttrIndex);
                }
            } else {
                data.instance(i).setMissing(newAttrIndex);
            }
        }
    }
}
