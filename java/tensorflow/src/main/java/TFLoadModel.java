import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.SavedModelBundle;

import java.util.List;
import java.io.*;
import org.apache.commons.io.*;

public class TFLoadModel {

	private static final String WEIGHT = "weight:0";
	private static final String BIAS = "bias:0";

	public static void main(String[] args) throws Exception {
		try (Graph graph = new Graph()) {

			String modelPath = "/home/skawtus/workspace/venv/tf2java_example/model/by_graph/linear";
			SavedModelBundle model = SavedModelBundle.load(modelPath, "train");
			List<Tensor<?>> outputs = null;

			try {

				outputs = model
						.session()
						.runner()
						.fetch(WEIGHT)
						.fetch(BIAS)
						.run();

			} catch (Exception e) {
				throw new RuntimeException(e.getMessage(), e);
			} finally {
				if (model != null) {
					model.close();
				}
				if (outputs != null) {
					for (Tensor output: outputs) {
						if(output != null) {
							System.out.println(output.toString()+ " " +output.floatValue());
							output.close();
						}
					}
				}
			}
		}
	}
}