package chapter3;

import weka.core.converters.CSVLoader;
import java.io.File;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.Filter;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.classifiers.evaluation.Evaluation;

import java.util.Random;

import javax.swing.JFrame;
import weka.gui.treevisualizer.TreeVisualizer;
import weka.gui.treevisualizer.PlaceNode2;

public class RegressionTask {
	public static void main(String args[]) throws Exception {
		System.out.println("Start running...");
		
		/**
		 * Load data
		 */
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator(",");
		loader.setSource(new File("data/ENB2012_data.csv"));
		Instances data = loader.getDataSet();
		
		/**
		 * Build regression models
		 */
		// set class index to Y1 (heating load)
		data.setClassIndex(data.numAttributes() - 2);
		// remove cooling load
		Remove remove = new Remove();
		remove.setOptions(new String[] { "-R", data.numAttributes() + "" });
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);
		
		// build regression model
		LinearRegression model = new LinearRegression();
		model.buildClassifier(data);
		System.out.println(model);
		
		// 10-fold cross validation
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 10, new Random(1), new Object[] {});
		System.out.println(eval.toSummaryString());
		double coef[] = model.coefficients();
		System.out.println();
		
		// build regression tree model
		M5P md5 = new M5P();
		md5.buildClassifier(data);
		System.out.println(md5);
		
		/*
		 * Visualize decision tree
		 */
		TreeVisualizer tv = new TreeVisualizer(null, md5.graph(),
				new PlaceNode2());
		JFrame frame = new javax.swing.JFrame("Tree Visualizer");
		frame.setSize(1600, 800);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(tv);
		frame.setVisible(true);
		tv.fitToScreen();
		
		// 10-fold cross validation
		eval.crossValidateModel(md5, data, 10, new Random(1), new Object[] {});
		System.out.println(eval.toSummaryString());
		System.out.println();
		
	}
}
