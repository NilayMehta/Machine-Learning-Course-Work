import java.util.ArrayList;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.ErrorMeasure;
import shared.DataSet;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.ga.StandardGeneticAlgorithm;
import opt.SimulatedAnnealing;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.example.NeuralNetworkOptimizationProblem;
import java.lang.StringBuilder;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedReader;
import java.util.Scanner;
import java.text.DecimalFormat;
import java.util.Arrays;

import shared.DataSet;
import shared.Instance;
import shared.filt.LabelSplitFilter;
import shared.reader.CSVDataSetReader;


public class ProjectRunner {
	
	
	static String[] datasetSizes = {"600", "1200", "1800", "2400", "3000"};
	private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
	private static ErrorMeasure measure = new SumOfSquaresError();
	private static DecimalFormat df = new DecimalFormat("0.000");
	/**
	 * @param args
	 * @throws Exception 
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException, Exception {
        randomrestartRHC();               //random restarts for randomized hill climbing
        SA();                             //Simulated Annealing
        GA();                             //Genetic Algorithm
        TravelingSalesman.main(null);     //Optimization Problem 1
        FlipFlop.main(null);              //Optimization Problem 2
        FourPeaks.main(null);             //Optimization Problem 3
    }
	
	
	
	private static void randomrestartRHC() throws Exception, IOException {
	      System.out.println("\nRandom restarting RHC");
	      int[] numberOfRestarts = {1, 2, 3, 5, 8, 10, 15, 20};
	      int[] iterations = {1000, 2000, 4000, 7000, 10000, 15000};

	    	for (String DSsize : datasetSizes) {
	    		for (int i : iterations) {
	    			
	    			DataSet set_train = null;
			    	DataSet set_test = null;
			  		try {
			  			set_train = (new CSVDataSetReader("datasets/train_HRAnalytics_" + DSsize + ".csv")).read();
			  			set_test = (new CSVDataSetReader("datasets/test_HRAnalytics_" + DSsize + ".csv")).read();
		
			  		} catch (Exception e1) {
			  			e1.printStackTrace();
			  		}
			  		(new LabelSplitFilter()).filter(set_train);
			  		(new LabelSplitFilter()).filter(set_test);
			    	  
			  		(new NeuralNetTest(i, set_train, set_test, Integer.parseInt(DSsize))).runRHC();
			  		
			        System.out.println("Training multiple NNs with RHC and sizes =" + DSsize);
			          
	    		}
	   
	      }
	  }
	
	
	private static void SA() throws Exception, IOException {
	      System.out.println("\nSA");
	      int[] iterations = {7000};
	      String[] datasetSize = {"3000"};

	    	for (String DSsize : datasetSize) {
	    		for (int i : iterations) {
	    			
	    			DataSet set_train = null;
			    	DataSet set_test = null;
			  		try {
			  			set_train = (new CSVDataSetReader("datasets/train_HRAnalytics_" + DSsize + ".csv")).read();
			  			set_test = (new CSVDataSetReader("datasets/test_HRAnalytics_" + DSsize + ".csv")).read();
		
			  		} catch (Exception e1) {
			  			e1.printStackTrace();
			  		}
			  		(new LabelSplitFilter()).filter(set_train);
			  		(new LabelSplitFilter()).filter(set_test);
			    	  
			  		(new NeuralNetTest(i, set_train, set_test, Integer.parseInt(DSsize))).runSA();
			  		
			        System.out.println("Training multiple NNs with SA and sizes =" + DSsize);   
	    		}
	      }
	  }
	
	private static void GA() throws Exception, IOException {
	      System.out.println("\nSA");
	      int[] iterations = {50};
	      String[] datasetSize = {"1800"};

	    	for (String DSsize : datasetSize) {
	    		for (int i : iterations) {
	    			
	    			DataSet set_train = null;
			    	DataSet set_test = null;
			  		try {
			  			set_train = (new CSVDataSetReader("datasets/train_HRAnalytics_" + DSsize + ".csv")).read();
			  			set_test = (new CSVDataSetReader("datasets/test_HRAnalytics_" + DSsize + ".csv")).read();
		
			  		} catch (Exception e1) {
			  			e1.printStackTrace();
			  		}
			  		(new LabelSplitFilter()).filter(set_train);
			  		(new LabelSplitFilter()).filter(set_test);
			    	  
			  		(new NeuralNetTest(i, set_train, set_test, Integer.parseInt(DSsize))).runGA();
			  		
			        System.out.println("Training multiple NNs with GA and sizes =" + DSsize);   
	    		}
	      }
	  }


}
