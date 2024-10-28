package tasks;

import weka.core.Instances;
import java.util.Arrays;
import java.util.List;

public class FullPipeline {
     private static final List<String> POLLUTANTS = Arrays.asList(
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
    );
    public static void main(String[] args) {

        try {

            processAndSave(
                "/home/cwreece/00.Projects/AQIndia/city_day.csv",
                "/home/cwreece/00.Projects/AQIndia/cleaned_city_day.csv",
                "Cleaned Data"
            );

            Instances featureEngineeredData = loadAndApplyFeatureEngineering(
                "/home/cwreece/00.Projects/AQIndia/cleaned_city_day.csv",
                "/home/cwreece/00.Projects/AQIndia/feature_engineered_city_day.csv"
            );

            Instances normalizedData = normalizeAndSave(
                featureEngineeredData,
                "/home/cwreece/00.Projects/AQIndia/normalized_city_day.csv"
            );
            Instances standardizedData = standardizeAndSave(
                featureEngineeredData,
                "/home/cwreece/00.Projects/AQIndia/standardized_city_day.csv"
            );

            Instances selectedNormalizedData = selectFeaturesAndSave(
                normalizedData,
                "/home/cwreece/00.Projects/AQIndia/selected_normalized_city_day.csv"
            );
            Instances selectedStandardizedData = selectFeaturesAndSave(
                standardizedData,
                "/home/cwreece/00.Projects/AQIndia/selected_standardized_city_day.csv"
            );
System.out.println("Max Heap Size: " + Runtime.getRuntime().maxMemory() / (1024 * 1024) + " MB");
            String reportPath = "/home/cwreece/00.Projects/AQIndia/model_training_report.csv";
            ModelTraining.initializeReport(reportPath);
            ModelTraining.trainAndEvaluateModels(selectedNormalizedData, "Normalized", reportPath);
            ModelTraining.trainAndEvaluateModels(selectedStandardizedData, "Standardized", reportPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Instances processAndSave(String inputPath, String outputPath, String stage) throws Exception {
        Instances data = AirQualityProcessor.process(inputPath);
        validate(data, stage);
        AirQualityProcessor.save(data, outputPath);
        System.out.println(stage + " saved to: " + outputPath);
        return data;
    }

    private static Instances loadAndApplyFeatureEngineering(String inputPath, String outputPath) throws Exception {
        Instances data = AirQualityProcessor.loadDataset(inputPath);
        AirQualityFeatureEngineering.apply(data, POLLUTANTS);
        validate(data, "Feature-Engineered Data");
        AirQualityProcessor.save(data, outputPath);
        return data;
    }

    private static Instances normalizeAndSave(Instances data, String outputPath) throws Exception {
        Instances normalizedData = DataNormalization.normalize(data);
        validate(normalizedData, "Normalized Data");
        AirQualityProcessor.save(normalizedData, outputPath);
        return normalizedData;
    }

    private static Instances standardizeAndSave(Instances data, String outputPath) throws Exception {
        Instances standardizedData = DataStandardization.standardize(data);
        validate(standardizedData, "Standardized Data");
        AirQualityProcessor.save(standardizedData, outputPath);
        return standardizedData;
    }

    private static Instances selectFeaturesAndSave(Instances data, String outputPath) throws Exception {
        Instances selectedData = FeatureSelector.selectFeatures(data);
        validate(selectedData, "Selected Features");
        AirQualityProcessor.save(selectedData, outputPath);
        return selectedData;
    }

    private static void validate(Instances data, String stage) {
        boolean isValid = isValid(data);
        if (isValid) {
            System.out.println(stage + " is valid with no missing values.");
        } else {
            System.err.println("Warning: Missing values found during " + stage + " stage.");
        }
    }

   private static boolean isValid(Instances data) {
    for (int i = 0; i < data.numAttributes(); i++) {
        String attributeName = data.attribute(i).name();

        if (attributeName.contains("_rolling_")) {
            int rollingWindow = getRollingWindow(attributeName);
            if (hasMissingValuesBeyondWindow(data, i, rollingWindow)) {
                System.err.println("Missing values in attribute: " + attributeName);
                return false;
            }
        } else if (data.attributeStats(i).missingCount > 0) {
            System.err.println("Missing values in attribute: " + attributeName);
            return false;
        }
    }
    return true;
}

private static int getRollingWindow(String attributeName) {
    String[] parts = attributeName.split("_");
    return Integer.parseInt(parts[2].replace("d", ""));
}

private static boolean hasMissingValuesBeyondWindow(Instances data, int attributeIndex, int windowSize) {
    for (int i = windowSize; i < data.numInstances(); i++) {
        if (data.instance(i).isMissing(attributeIndex)) {
            return true;
        }
    }
    return false;
}
}
