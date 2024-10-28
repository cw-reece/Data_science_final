package tasks;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RandomCommittee;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;
import java.util.concurrent.*;
public class ModelTraining {

    public static void initializeReport(String reportPath) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(reportPath))) {
            writer.println("Dataset,Model,Accuracy/RMSE,Precision,Recall,F1Score");
            System.out.println("Report initialized at: " + reportPath);
        }
    }


    public static void trainAndEvaluateModels(Instances data, String datasetType, String reportPath) throws Exception {

        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        if (data.numInstances() == 0) {
            System.err.println("Error: No instances found in the dataset.");
            return;
        }

        Classifier[] models = {
                new ZeroR(),
                new IBk(),
                new RandomForest(),
                new M5P(),
                new RandomCommittee()
        };

        try (PrintWriter writer = new PrintWriter(new FileWriter(reportPath, true))) {
            for (Classifier model : models) {
                evaluateModelWithTimeout(model, data, datasetType, writer, reportPath);
            }
        } catch (IOException e) {
            System.err.println("Error writing to report: " + e.getMessage());
            e.printStackTrace();
        }
    }


    private static void evaluateModelWithTimeout(Classifier model, Instances data, String datasetType,
                                                 PrintWriter writer, String reportPath) {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<?> future = executor.submit(() -> {
            try {
                System.out.println("Evaluating: " + model.getClass().getSimpleName());

                Evaluation eval = new Evaluation(data);

                long start = System.currentTimeMillis();
                eval.crossValidateModel(model, data, 3, new Random(1));
                long end = System.currentTimeMillis();
                System.out.println(model.getClass().getSimpleName() +
                        " completed in " + (end - start) / 1000 + " seconds.");

                String result;
                if (data.classAttribute().isNominal()) {
                    System.out.println("Performing classification evaluation...");
                    result = String.format(
                            "%s,%s,%.2f,%.4f,%.4f,%.4f,%.4f",
                            datasetType, model.getClass().getSimpleName(),
                            eval.pctCorrect(), eval.rootMeanSquaredError(),
                            eval.weightedPrecision(), eval.weightedRecall(), eval.weightedFMeasure()
                    );
                } else if (data.classAttribute().isNumeric()) {
                    System.out.println("Performing regression evaluation...");
                    result = String.format(
                            "%s,%s,%.4f,%.4f,%.4f",
                            datasetType, model.getClass().getSimpleName(),
                            eval.meanAbsoluteError(), eval.rootMeanSquaredError(), eval.correlationCoefficient()
                    );
                } else {
                    throw new IllegalArgumentException("Unsupported class attribute type.");
                }

                System.out.println(result);
                writer.println(result);
                writer.flush();

            } catch (Exception e) {
                System.err.println("Error evaluating model " + model.getClass().getSimpleName() + ": " + e.getMessage());
                e.printStackTrace();
                writer.println(datasetType + "," + model.getClass().getSimpleName() + ",ERROR");
            }
        });
        try {

            future.get(5, TimeUnit.MINUTES);
        } catch (TimeoutException e) {
            System.err.println("Model " + model.getClass().getSimpleName() + " evaluation timed out.");
            writer.println(datasetType + "," + model.getClass().getSimpleName() + ",TIMEOUT");
            future.cancel(true);
        } catch (Exception e) {
            System.err.println("Error executing model: " + e.getMessage());
            e.printStackTrace();
        } finally {
            executor.shutdown();
        }
    }
}
