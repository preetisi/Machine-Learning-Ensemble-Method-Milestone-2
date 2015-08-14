import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RunClassifiers {
	final static String[] DATASETS = { "anneal", "audiology", "autos", "balance-scale",
			"breast-cancer", "colic", "credit-a", "diabetes", "glass", "heart-c", "hepatitis",
			"hypothyroid" };
	final static Class<?>[] CLASSIFIERS = { SMO.class, MultilayerPerceptron.class, AdaBoostM1.class };
	final static Class<? extends Classifier> NB_CLASSIFIER = NaiveBayes.class;

	final static String[] EXAMPLE_DATASET = { "anneal" };
	final static Class<?>[] EXAMPLE_CLASSIFIER = { SMO.class };

	public static void main(String[] args) throws Exception {
		if (args.length < 1) {
			System.err.println("Usage: RunClassifiers <data dir>");
		} else {
			runClassifiers(args[0], DATASETS, CLASSIFIERS);
		}
	}

	static String formatClassiferName(String fullClassifierName) {
		String[] nameParts = fullClassifierName.split("\\.");
		return nameParts[nameParts.length - 1];
	}

	static void runExampleClassifier(String dataDir) throws Exception {
		runClassifiers(dataDir, EXAMPLE_DATASET, EXAMPLE_CLASSIFIER);
	}

	static void runClassifiers(String dataDir, String[] dataSets, Class<?>[] classifiers)
			throws Exception {
		List<Double> errorRatios = new ArrayList<Double>(dataSets.length);
		for (String dataSet : dataSets) {
//			System.out.printf("Running dataset '%s'\n", dataSet);
			Instances trainingData = getTrainingInstances(dataDir, dataSet);
			Instances testingData = getTestingInstances(dataDir, dataSet);

			// Each classifier will be run on the dataset and results stored.
			List<ClassifierResult> classifierResults = new LinkedList<ClassifierResult>();

			for (Class<?> classifier : classifiers) {
//				System.out.printf("\tRunning classifier: %s...\n", classifier.getSimpleName());

				ClassifierResult result = runClassifier(trainingData, testingData, classifier);
				classifierResults.add(result);
			}

			ClassifierResult nbResult = runClassifier(trainingData, testingData, NB_CLASSIFIER);
			String modelOutputPath = getModelOutputPath(dataDir, dataSet);
			double minErrorRatio = outputMinResult(classifierResults, nbResult, modelOutputPath);
			errorRatios.add(minErrorRatio);
		}

		outputSummaryResults(errorRatios);
	}

	static Instances getTrainingInstances(String dataDir, String dataSet) throws Exception {
		return getInstances(dataDir, dataSet, true/* isTraining */);
	}

	static Instances getTestingInstances(String dataDir, String dataSet) throws Exception {
		return getInstances(dataDir, dataSet, false/* isTraining */);
	}

	/**
	 * Returns the instances for a dataset.
	 * 
	 * @param dataDir
	 *            the root directory containing the dataset arff file
	 * @param dataSet
	 *            the name of the dataset
	 * @param isTraining
	 *            whether this is a training or testing dataset.
	 */
	static Instances getInstances(String dataDir, String dataSet, boolean isTraining)
			throws Exception {
		String datasetRootName = dataDir + "/" + dataSet;
		String trainingDataPath = datasetRootName + (isTraining ? "_train.arff" : "_test.arff");

		DataSource inputData = new DataSource(trainingDataPath);

		Instances instances = inputData.getDataSet();

		if (instances.classIndex() == -1) {
			instances.setClassIndex(instances.numAttributes() - 1);
		}

		return instances;
	}

	static String getModelOutputPath(String dataDir, String dataSet) {
		String datasetRootName = dataDir + "/" + dataSet;
		return datasetRootName + ".model";
	}

	/**
	 * Prints the average and max error ratios, and outputs the model
	 * corresponding to the minimum error rate.
	 */
	static double outputMinResult(Collection<ClassifierResult> classifierResults,
			ClassifierResult nbResult, String outputModelPath) throws FileNotFoundException,
			IOException {
		// Find the result with the min error.
		ClassifierResult minErrorResult = Collections.min(classifierResults);

		// Output the results.
		// System.out.println("Max: " + maxErrorRatio);
		// System.out.println("Avg: " + meanErrorRatio);
		double minErrorRatio = minErrorResult.errorRate / nbResult.errorRate;
		System.out.println("Min error ratio: " + minErrorRatio);
        System.out.println("Outputting the model with min error to " + outputModelPath);
		minErrorResult.outputModel(outputModelPath);
		return minErrorRatio;
	}

	static void outputSummaryResults(Collection<Double> minErrorRatios) {
		System.out.println("Max error ratio: " + Collections.max(minErrorRatios));

		// Find the mean error rate.
		double totalErrorRatio = 0;
		for (Double errorRatio : minErrorRatios) {
			totalErrorRatio += errorRatio;
		}

		double meanErrorRatio = (totalErrorRatio / minErrorRatios.size());
		System.out.println("Mean error ratio: " + meanErrorRatio);
	}

	static ClassifierResult runClassifier(Instances trainingData, Instances testingData,
			Class<?> classifier) throws Exception {
		Classifier model = Classifier.forName(classifier.getName(), null);
		model.buildClassifier(trainingData);

		double modelError = getError(model, trainingData, testingData);
		return new ClassifierResult(modelError, model);
	}

	static double getError(Classifier model, Instances trainData, Instances testData)
			throws Exception {
		// Evaluate on the test dataset.
		Evaluation eval = new Evaluation(trainData);
		eval.evaluateModel(model, testData);
		return eval.errorRate();
	}

}
