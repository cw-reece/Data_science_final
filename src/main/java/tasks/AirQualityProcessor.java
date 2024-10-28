package tasks;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import java.io.File;
import java.io.IOException;

public class AirQualityProcessor {
    public static Instances process(String filePath) throws Exception {
        Instances data = loadDataset(filePath);  // Step 1: Load data
        data = replaceMissingValues(data);       // Step 2: Handle missing values

        // Optional: Validate the dataset to catch any remaining issues
        validateData(data);

        return data;  // Return cleaned dataset
    }

    // Loads the dataset from a CSV file
    static Instances loadDataset(String filePath) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filePath));
        Instances data = loader.getDataSet();
        System.out.println("Dataset loaded. Total instances: " + data.numInstances());
        return data;
    }

    // Replaces missing values using the ReplaceMissingValues filter
    private static Instances replaceMissingValues(Instances data) throws Exception {
        ReplaceMissingValues replaceFilter = new ReplaceMissingValues();
        replaceFilter.setInputFormat(data);
        Instances cleanedData = Filter.useFilter(data, replaceFilter);
        System.out.println("Missing values replaced.");
        return cleanedData;
    }

    // Optional: Validate the dataset to ensure no critical issues remain
    private static void validateData(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attributeStats(i).missingCount > 0) {
                System.err.println("Warning: Missing values remain in attribute: " + data.attribute(i).name());
            }
        }
    }

    // Saves the cleaned dataset to a CSV file
    public static void save(Instances data, String outputPath) throws IOException {
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
        System.out.println("Dataset saved to: " + outputPath);
    }
}
