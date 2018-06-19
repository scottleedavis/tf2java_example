import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import java.io.*;
import org.apache.commons.io.*;

public class TFLoadModel {
  public static void main(String[] args) throws Exception {
    try (Graph graph = new Graph()) {
      final String value = "Hello from " + TensorFlow.version();
      System.out.println(value);
      byte[] pbfile = FileUtils.readFileToByteArray(new File("/home/skawtus/workspace/venv/tf2java_example/model/by_graph/linear/saved_model.pb"));

      // byte[] pbfile = IOUtils.toByteArray(new FileInputStream(new File("/home/skawtus/workspace/venv/tf2java_example/model/by_graph/linear/saved_model.pb")));
      // System.out.println(new String(pbfile));
      graph.importGraphDef(pbfile);


      // // Construct the computation graph with a single operation, a constant
      // // named "MyConst" with a value "value".
      // try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
      //   // The Java API doesn't yet include convenience functions for adding operations.
      //   g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
      // }

      // // Execute the "MyConst" operation in a Session.
      // try (Session s = new Session(g);
      //      Tensor output = s.runner().fetch("MyConst").run().get(0)) {
      //   System.out.println(new String(output.bytesValue(), "UTF-8"));
      // }
    }
  }
}