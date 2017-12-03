package chapter3;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.core.Instances;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;

public class ClusteringTask {
	public static void main(String args[]) throws Exception {
		// load data
		Instances data = new Instances(new BufferedReader(new FileReader("data/bank-data.arff")));
		
		EM model = new EM();
		model.buildClusterer(data);
		System.out.println(model);
		
		double evalResult = ClusterEvaluation.crossValidateModel(model, data, 10, new Random(1));
		System.out.println(evalResult);
	}
}
