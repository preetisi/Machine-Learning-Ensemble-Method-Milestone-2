import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;

/**
 * Encapsulates the result of building and evaluating a model.
 */
public class ClassifierResult implements Comparable<ClassifierResult> {
	public final double errorRate;
	private final Classifier model;

	public ClassifierResult(double errorRate, Classifier model) {
		this.errorRate = errorRate;
		this.model = model;
	}

	/**
	 * Outputs the model to the given path.
	 */
	public void outputModel(String modelOutputPath) throws FileNotFoundException, IOException {
		// serialize weka output
		// instead of 4 lines just write this line to serialize
		// source: http://weka.wikispaces.com/Serialization

		// weka.core.SerializationHelper.write("/some/where/j48.model", cls);

		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelOutputPath));
		oos.writeObject(model);
		oos.flush();
		oos.close();
	}

	public String getName() {
		return model.getClass().getSimpleName();
	}

	@Override
	public int compareTo(ClassifierResult o) {
		return Double.valueOf(errorRate).compareTo(o.errorRate);
	}
}
